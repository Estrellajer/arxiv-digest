/**
 * 飞书事件订阅 Webhook（Cloudflare Worker）
 *
 * 流程：飞书 IM 消息 → 本 Worker（秒回 200）→ 解析 arxiv 链接与关键词
 *      → 触发 GitHub repository_dispatch（event_type: arxiv-paper）
 *      → GitHub Actions 跑 OCR + 抽取 → 飞书 app 把结果卡片推回原会话。
 *
 * 关键词：消息含「精读」走精读(reading)，否则默认实验配置(setup)。
 *
 * 需要配置的 Secrets / Vars（见 wrangler.toml）：
 *   GITHUB_TOKEN              必填，具备 repo 的 repository_dispatch 权限（PAT 或细粒度 token）
 *   GITHUB_REPO              必填，形如 "owner/repo"
 *   FEISHU_VERIFICATION_TOKEN 可选，事件订阅的 Verification Token，校验来源
 *   FEISHU_ENCRYPT_KEY       可选，开启了事件加密时填，用于解密
 *   FEISHU_APP_ID            可选，填了则发送「正在解析…」即时回执
 *   FEISHU_APP_SECRET        可选，同上
 */

const ARXIV_RE = /(?:arxiv\.org\/(?:abs|pdf)\/)?(\d{4}\.\d{4,5})(v\d+)?(?:\.pdf)?/i;

export default {
  async fetch(request, env, ctx) {
    if (request.method !== "POST") {
      return new Response("ok");
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return new Response("bad request", { status: 400 });
    }

    // 事件加密：body 为 { "encrypt": "..." }
    if (payload.encrypt) {
      if (!env.FEISHU_ENCRYPT_KEY) {
        return new Response("encrypt key not configured", { status: 500 });
      }
      try {
        payload = JSON.parse(await decryptFeishu(payload.encrypt, env.FEISHU_ENCRYPT_KEY));
      } catch (e) {
        return new Response("decrypt failed", { status: 400 });
      }
    }

    // URL 验证（配置事件订阅地址时飞书会发一次）
    if (payload.type === "url_verification") {
      return jsonResponse({ challenge: payload.challenge });
    }

    // 校验来源 token（v2 在 header.token，v1 在 body.token）
    const token = payload.header?.token || payload.token;
    if (env.FEISHU_VERIFICATION_TOKEN && token !== env.FEISHU_VERIFICATION_TOKEN) {
      return new Response("forbidden", { status: 403 });
    }

    const eventType = payload.header?.event_type || payload.event?.type;
    if (eventType === "im.message.receive_v1") {
      const message = payload.event?.message || {};
      const chatId = message.chat_id;
      const text = parseText(message);
      const url = extractArxivUrl(text);
      if (url && chatId) {
        const task = /精读/.test(text) ? "reading" : "setup";
        // 重活异步做，先把 200 还给飞书，避免 3 秒超时重推
        ctx.waitUntil(handlePaper(env, { url, task, chatId }));
      }
    }

    return jsonResponse({ code: 0 });
  },
};

function parseText(message) {
  if (message.message_type !== "text") return "";
  try {
    return JSON.parse(message.content || "{}").text || "";
  } catch {
    return "";
  }
}

function extractArxivUrl(text) {
  if (!text) return "";
  const m = text.match(ARXIV_RE);
  if (!m) return "";
  // 统一回传 abs 链接；下游脚本会再归一化为 pdf 交给 MinerU
  return `https://arxiv.org/abs/${m[1]}`;
}

async function handlePaper(env, { url, task, chatId }) {
  if (env.FEISHU_APP_ID && env.FEISHU_APP_SECRET) {
    const label = task === "reading" ? "精读" : "实验配置";
    await sendAck(env, chatId, `🔬 已收到，正在解析${label}：${url}\n约需 1 分钟，完成后推送结果。`).catch(() => {});
  }
  await dispatchGitHub(env, { url, task, chatId });
}

async function dispatchGitHub(env, clientPayload) {
  const resp = await fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "Content-Type": "application/json",
      "User-Agent": "arxiv-feishu-bot",
    },
    body: JSON.stringify({ event_type: "arxiv-paper", client_payload: clientPayload }),
  });
  if (!resp.ok) {
    console.log(`GitHub dispatch failed: ${resp.status} ${await resp.text()}`);
  }
}

// ── Feishu app 即时回执 ────────────────────────────────────────────────
async function sendAck(env, chatId, content) {
  const token = await getTenantToken(env);
  await fetch("https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      receive_id: chatId,
      msg_type: "text",
      content: JSON.stringify({ text: content }),
    }),
  });
}

async function getTenantToken(env) {
  const resp = await fetch(
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ app_id: env.FEISHU_APP_ID, app_secret: env.FEISHU_APP_SECRET }),
    }
  );
  const data = await resp.json();
  return data.tenant_access_token;
}

// ── 工具 ────────────────────────────────────────────────────────────
function jsonResponse(obj) {
  return new Response(JSON.stringify(obj), {
    headers: { "Content-Type": "application/json" },
  });
}

// 飞书事件加密：AES-256-CBC，key = SHA256(encryptKey)，密文 base64，前 16 字节为 IV
async function decryptFeishu(encrypt, encryptKey) {
  const keyBytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(encryptKey));
  const data = base64ToBytes(encrypt);
  const iv = data.slice(0, 16);
  const ciphertext = data.slice(16);
  const cryptoKey = await crypto.subtle.importKey("raw", keyBytes, { name: "AES-CBC" }, false, ["decrypt"]);
  const plain = await crypto.subtle.decrypt({ name: "AES-CBC", iv }, cryptoKey, ciphertext);
  return new TextDecoder().decode(plain);
}

function base64ToBytes(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
