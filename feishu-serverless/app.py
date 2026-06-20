"""
飞书事件 webhook（Flask Web 函数）。

可作为「Web 函数 / HTTP 函数」部署在国内 serverless（阿里云函数计算、腾讯云 SCF 等），
飞书国内入口可稳定访问，规避 Cloudflare 在国内不可达的问题。

需要的环境变量：
  GITHUB_TOKEN               必填，具备 repository_dispatch 权限
  GITHUB_REPO                必填，形如 "owner/repo"
  FEISHU_VERIFICATION_TOKEN  可选，事件订阅 Verification Token（校验来源）
  FEISHU_ENCRYPT_KEY         可选，开启事件加密时填（需 cryptography 依赖）
  FEISHU_APP_ID              可选，填了则发送「正在解析…」即时回执
  FEISHU_APP_SECRET          可选，同上

本地调试：python app.py  然后用 ngrok / 内网穿透临时暴露（生产请部署到国内 serverless）。
"""

import json
import os
import urllib.request

from flask import Flask, request, jsonify

import core

app = Flask(__name__)


def dispatch_github(url: str, task: str, chat_id: str):
    body = json.dumps({
        "event_type": "arxiv-paper",
        "client_payload": {"url": url, "task": task, "chat_id": chat_id},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPO']}/dispatches",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "arxiv-feishu-bot",
        },
    )
    urllib.request.urlopen(req, timeout=10)


def send_ack(chat_id: str, task: str, url: str):
    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")
    if not app_id or not app_secret:
        return
    token_req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(token_req, timeout=10) as resp:
        token = json.loads(resp.read()).get("tenant_access_token")
    if not token:
        return
    label = "精读" if task == "reading" else "实验配置"
    msg_req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        data=json.dumps({
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": f"🔬 已收到，正在解析{label}：{url}\n约需 1 分钟，完成后推送结果。"}),
        }).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    urllib.request.urlopen(msg_req, timeout=10)


@app.route("/", methods=["GET", "POST"])
@app.route("/feishu", methods=["GET", "POST"])
def feishu():
    if request.method == "GET":
        return "ok"
    status, body = core.handle_event(
        request.get_data(),
        dict(os.environ),
        dispatch=dispatch_github,
        ack=send_ack,
    )
    return jsonify(body), status


if __name__ == "__main__":
    # 阿里云 FC / 腾讯云 SCF 的 Web 函数默认监听 9000 端口
    port = int(os.environ.get("PORT") or os.environ.get("FC_SERVER_PORT") or "9000")
    app.run(host="0.0.0.0", port=port)
