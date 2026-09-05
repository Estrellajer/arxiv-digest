import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import utils
import digest
import paper_context
import urllib.error


class WorkflowContractTests(unittest.TestCase):
    def test_repository_contains_only_the_supported_scf_receiver(self):
        self.assertFalse((ROOT / "feishu-worker").exists())
        self.assertFalse((ROOT / "feishu-serverless/app.py").exists())
        self.assertFalse((ROOT / "CLAUDE.md").exists())
        self.assertTrue((ROOT / "feishu-serverless/index.py").exists())
        self.assertTrue((ROOT / "feishu-serverless/core.py").exists())

    def test_readme_documents_reuse_and_paddleocr_migration(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("为什么从 PaddleOCR 改为 MinerU", readme)
        self.assertIn("10004 文件格式不支持", readme)
        self.assertIn("Contents: Read and write", readme)
        self.assertIn("index.main_handler", readme)
    def test_paper_analysis_has_issue_permission_and_enables_ocr(self):
        workflow = (ROOT / ".github/workflows/paper-analysis.yml").read_text(encoding="utf-8")

        self.assertIn("issues: write", workflow)
        self.assertIn("--use-ocr --push-to-feishu", workflow)
        self.assertNotIn("FEISHU_WEBHOOK: ${{ secrets.FEISHU_WEBHOOK }}", workflow)
        self.assertIn("d.get('quick_take'", workflow)
        self.assertNotIn("d.get('core_claim'", workflow)
        # 文档解析使用 MinerU；旧 OCR Token 不再需要
        self.assertIn("MINERU_TOKEN: ${{ secrets.MINERU_TOKEN }}", workflow)
        self.assertNotIn("OCR_API_TOKEN", workflow)

    def test_daily_digest_uses_mineru_token_and_uploads_artifact(self):
        workflow = (ROOT / ".github/workflows/daily-digest.yml").read_text(encoding="utf-8")
        self.assertIn("output/digest_analysis.json", workflow)
        self.assertIn("MINERU_TOKEN: ${{ secrets.MINERU_TOKEN }}", workflow)
        self.assertIn("cron: '17 22 * * *'", workflow)
        self.assertNotIn("OCR_API_TOKEN", workflow)

    def test_experiment_setup_workflow_triggers_and_pushes(self):
        workflow = (ROOT / ".github/workflows/experiment-setup.yml").read_text(encoding="utf-8")
        self.assertIn("issues: write", workflow)
        self.assertIn("[实验配置]", workflow)
        self.assertIn("scripts/extract_setup.py", workflow)
        self.assertIn("--push-to-feishu", workflow)
        self.assertIn("MINERU_TOKEN: ${{ secrets.MINERU_TOKEN }}", workflow)

    def test_feishu_dispatch_workflow_routes_link_to_chat(self):
        workflow = (ROOT / ".github/workflows/feishu-dispatch.yml").read_text(encoding="utf-8")
        # 由飞书 webhook 通过 repository_dispatch 触发
        self.assertIn("repository_dispatch", workflow)
        self.assertIn("arxiv-paper", workflow)
        # 结果推回发消息的会话
        self.assertIn("github.event.client_payload.chat_id || github.event.inputs.chat_id", workflow)
        # 关键词分流到精读 / 实验配置
        self.assertIn("scripts/extract_setup.py", workflow)
        self.assertIn("scripts/reading.py", workflow)
        self.assertIn("MINERU_TOKEN: ${{ secrets.MINERU_TOKEN }}", workflow)
        # 可从 Actions 页面重放 payload，独立排除飞书 webhook 故障
        self.assertIn("workflow_dispatch", workflow)
        self.assertIn("github.event.inputs.arxiv_url", workflow)
        self.assertIn("github.event.inputs.chat_id", workflow)


class FeishuWebhookCoreTests(unittest.TestCase):
    """国内 serverless webhook 的纯逻辑（feishu-serverless/core.py）。"""

    @staticmethod
    def _core():
        import importlib.util

        path = ROOT / "feishu-serverless" / "core.py"
        spec = importlib.util.spec_from_file_location("feishu_core", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_url_verification_echoes_challenge(self):
        core = self._core()
        status, body = core.handle_event(
            {"type": "url_verification", "challenge": "abc"}, {}, dispatch=lambda *a: None
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["challenge"], "abc")

    def test_message_dispatches_setup_by_default(self):
        core = self._core()
        calls = []
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {"message": {
                "chat_id": "oc_1", "message_type": "text",
                "content": json.dumps({"text": "看下 https://arxiv.org/abs/2210.03629"}),
            }},
        }
        status, body = core.handle_event(payload, {}, dispatch=lambda *a: calls.append(a))
        self.assertEqual(calls, [("https://arxiv.org/abs/2210.03629", "setup", "oc_1")])
        self.assertEqual(status, 200)
        self.assertTrue(body["dispatch_attempted"])
        self.assertTrue(body["dispatch_ok"])

    def test_message_keyword_routes_to_reading_and_bare_id(self):
        core = self._core()
        calls = []
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {"message": {
                "chat_id": "oc_2", "message_type": "text",
                "content": json.dumps({"text": "精读 2210.03629"}),
            }},
        }
        core.handle_event(payload, {}, dispatch=lambda *a: calls.append(a))
        self.assertEqual(calls, [("https://arxiv.org/abs/2210.03629", "reading", "oc_2")])

    def test_token_mismatch_is_forbidden(self):
        core = self._core()
        status, _ = core.handle_event(
            {"token": "wrong", "header": {"event_type": "im.message.receive_v1"}},
            {"FEISHU_VERIFICATION_TOKEN": "right"},
            dispatch=lambda *a: None,
        )
        self.assertEqual(status, 403)

    def test_dispatch_failure_is_observable(self):
        core = self._core()
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {"message": {
                "chat_id": "oc_1", "message_type": "text",
                "content": json.dumps({"text": "2210.03629"}),
            }},
        }

        def fail(*_args):
            raise RuntimeError("GitHub dispatch returned HTTP 403")

        status, body = core.handle_event(payload, {}, dispatch=fail)
        self.assertEqual(status, 502)
        self.assertTrue(body["dispatch_attempted"])
        self.assertFalse(body["dispatch_ok"])
        self.assertIn("403", body["error"])

    @staticmethod
    def _scf_index():
        import importlib.util

        d = ROOT / "feishu-serverless"
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        spec = importlib.util.spec_from_file_location("scf_index", d / "index.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_scf_handler_echoes_challenge(self):
        index = self._scf_index()
        event = {"httpMethod": "POST", "body": json.dumps({"type": "url_verification", "challenge": "xyz"})}
        resp = index.main_handler(event, None)
        self.assertEqual(resp["statusCode"], 200)
        self.assertIn("xyz", resp["body"])

    def test_scf_handler_dispatches_message(self):
        index = self._scf_index()
        event = {
            "httpMethod": "POST",
            "body": json.dumps({
                "header": {"event_type": "im.message.receive_v1"},
                "event": {"message": {
                    "chat_id": "oc_9", "message_type": "text",
                    "content": json.dumps({"text": "https://arxiv.org/abs/2210.03629"}),
                }},
            }),
        }
        with patch.object(index, "_dispatch") as disp:
            resp = index.main_handler(event, None)
        self.assertEqual(resp["statusCode"], 200)
        disp.assert_called_once_with("https://arxiv.org/abs/2210.03629", "setup", "oc_9")


class FeishuAppTests(unittest.TestCase):
    @patch("utils.httpx.post")
    @patch("utils._get_feishu_tenant_token", return_value="tenant-token")
    def test_send_defaults_to_chat_id(self, _token, post):
        post.return_value.json.return_value = {"code": 0}
        env = {
            "FEISHU_APP_ID": "app-id",
            "FEISHU_APP_SECRET": "app-secret",
            "FEISHU_RECEIVE_ID_TYPE": "",
        }

        with patch.dict(os.environ, env, clear=True):
            sent = utils.send_feishu_message("chat-id", "interactive", "{}")

        self.assertTrue(sent)
        self.assertIn("receive_id_type=chat_id", post.call_args.args[0])
        self.assertEqual(post.call_args.kwargs["json"]["receive_id"], "chat-id")


class InstitutionAndEvidenceTests(unittest.TestCase):
    def test_matches_company_and_us_china_universities(self):
        text = "Meta FAIR, Carnegie Mellon University, 清华大学 and an unknown lab"
        names = {item["name"] for item in paper_context.match_institutions(text)}
        self.assertEqual(names, {"Meta AI", "Carnegie Mellon University", "Tsinghua University"})

    def test_short_alias_does_not_match_inside_word(self):
        names = {item["name"] for item in paper_context.match_institutions("community and opportunity")}
        self.assertNotIn("Massachusetts Institute of Technology", names)
        self.assertNotIn("Nanyang Technological University", names)

    def test_bonus_requires_relevance_and_never_stacks(self):
        institutions = [{"name": "Meta AI"}, {"name": "Stanford University"}]
        self.assertEqual(paper_context.apply_institution_bonus(0.49, institutions), (0.49, 0.0))
        self.assertEqual(paper_context.apply_institution_bonus(0.70, institutions), (0.78, 0.08))

    def test_extracts_only_supported_evidence_urls(self):
        text = "Model https://huggingface.co/org/m and code https://github.com/org/repo.git and project https://example.com/project."
        self.assertEqual(paper_context.extract_evidence_urls(text), ["https://github.com/org/repo", "https://huggingface.co/org/m"])

    def test_extracts_numbered_markdown_experiment_section(self):
        ocr = {"markdown": "# 2. Method\nmethod\n# 3. Evaluation\nTable 1 result 88.0\n## 3.1 Ablation\nminus x\n# 4. Conclusion\ndone"}
        section = utils.extract_experiment_section(ocr)
        self.assertIn("Table 1 result 88.0", section)
        self.assertIn("minus x", section)
        self.assertNotIn("done", section)

    def test_company_name_in_abstract_is_not_an_affiliation(self):
        page = "# A Study of Qwen\nAlice, Bob\nSmall Research Lab\n\n# Abstract\nWe compare Meta and Qwen."
        region = paper_context.extract_affiliation_region(page)
        names = {item["name"] for item in paper_context.match_institutions(region)}
        self.assertNotIn("Meta AI", names)
        self.assertNotIn("Alibaba Qwen", names)

    @patch("digest.build_ocr_evidence")
    @patch("digest.ocr_arxiv_pdf")
    def test_digest_ocr_failure_degrades_without_aborting_batch(self, ocr, build):
        ocr.side_effect = RuntimeError("offline")
        paper = {"arxiv_id": "1", "abstract_url": "https://arxiv.org/abs/1", "content_score": 0.7}
        result = digest.enrich_with_ocr([paper], candidate_limit=1)
        self.assertEqual(result[0]["ocr_status"], "failed")
        self.assertEqual(result[0]["final_score"], 0.7)
        build.assert_not_called()

    @patch("digest.send_feishu_card", return_value=True)
    @patch("digest.fetch_arxiv_papers", return_value=[])
    def test_digest_notifies_feishu_when_no_papers_found(self, _fetch, send):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                digest.run_digest(
                    categories=["cs.CL"],
                    keywords=["agent"],
                    webhook_url="https://example.test/webhook",
                )
                analysis = json.loads(Path("output/digest_analysis.json").read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)

        self.assertEqual(analysis, [])
        self.assertTrue(send.called)
        card = send.call_args.args[1]
        self.assertIn("今日没有抓到", json.dumps(card, ensure_ascii=False))

    @patch("digest.send_feishu_card", return_value=True)
    @patch("digest.enrich_with_ocr")
    @patch("digest.score_papers")
    @patch("digest.fetch_arxiv_papers")
    def test_digest_notifies_feishu_when_no_paper_passes_threshold(self, fetch, score, enrich, send):
        paper = {
            "title": "Low score paper",
            "summary": "abstract",
            "arxiv_id": "1",
            "abstract_url": "https://arxiv.org/abs/1",
            "authors": [],
            "published": "2026-06-27T00:00:00Z",
        }
        scored = {**paper, "content_score": 0.1, "score": 0.1, "final_score": 0.1}
        fetch.return_value = [paper]
        score.return_value = [scored]
        enrich.return_value = [scored]

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                digest.run_digest(
                    categories=["cs.CL"],
                    keywords=["agent"],
                    threshold=0.75,
                    webhook_url="https://example.test/webhook",
                )
                analysis = json.loads(Path("output/digest_analysis.json").read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)

        self.assertEqual(analysis[0]["final_score"], 0.1)
        self.assertTrue(send.called)
        card = send.call_args.args[1]
        self.assertIn("今日没有论文达到推送阈值", json.dumps(card, ensure_ascii=False))

    def test_digest_card_exposes_score_components_and_coverage(self):
        paper = {
            "title": "Test", "abstract_url": "https://arxiv.org/abs/1", "authors": [],
            "content_score": 0.7, "institution_bonus": 0.08, "final_score": 0.78,
            "recognized_institutions": [{"name": "Meta AI"}], "code_url": "https://github.com/a/b",
            "ocr_status": "success", "decision": {"research_question": "Q", "core_method": "M", "key_experiment": "E", "recommendation": "R", "risk": "X"},
            "digest_cn": "M",
        }
        card = utils.build_digest_card([paper])
        content = card["elements"][1]["content"]
        self.assertIn("Meta AI", content)
        self.assertIn("内容 0.70 + 机构 0.08 = 0.78", content)
        self.assertIn("OCR 首页+实验", content)

    @patch("digest.llm_chat")
    def test_digest_accepts_only_grounded_code_selection(self, chat):
        chat.return_value = '{"core_method":"M","code_url":"https://evil.example/repo"}'
        paper = {"arxiv_id": "1", "title": "T", "summary": "S", "ocr_evidence": {"code_urls": ["https://github.com/a/b"]}, "code_url": "https://github.com/a/b"}
        result = digest.generate_digests([paper], model="test", max_workers=1)[0]
        self.assertEqual(result["code_url"], "https://github.com/a/b")

    def test_digest_card_offers_setup_extraction_button(self):
        paper = {
            "title": "Test", "abstract_url": "https://arxiv.org/abs/1", "authors": [],
            "content_score": 0.7, "institution_bonus": 0.0, "final_score": 0.7,
            "recognized_institutions": [], "code_url": None, "ocr_status": "success",
            "decision": {}, "digest_cn": "M",
        }
        card = utils.build_digest_card([paper])
        blob = json.dumps(card, ensure_ascii=False)
        self.assertIn("实验配置", blob)  # 新增的实验配置抽取按钮
        self.assertIn("精读", blob)

    def test_digest_issue_buttons_follow_the_fork_repository(self):
        paper = {
            "title": "Test", "abstract_url": "https://arxiv.org/abs/1", "authors": [],
            "content_score": 0.7, "institution_bonus": 0.0, "final_score": 0.7,
            "recognized_institutions": [], "code_url": None, "ocr_status": "success",
            "decision": {}, "digest_cn": "M",
        }
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "someone/fork"}, clear=False):
            card = utils.build_digest_card([paper])
        blob = json.dumps(card, ensure_ascii=False)
        self.assertIn("github.com/someone/fork/issues/new", blob)
        self.assertNotIn("github.com/Estrellajer/arxiv-digest/issues/new", blob)


class ExperimentSetupTests(unittest.TestCase):
    def test_setup_card_renders_all_sections(self):
        setup = {
            "title": "ReAct", "input_coverage": "MinerU 文档解析（前20页）",
            "datasets": [{"name": "GSM8K", "split": "test", "size": "1319"}],
            "model": {"name": "PaLM", "architecture": "decoder", "params": "540B", "init_weights": "pretrained"},
            "hyperparameters": {"learning_rate": "1e-5", "batch_size": "32", "epochs_or_steps": "3", "optimizer": "Adam"},
            "hardware": {"accelerator": "A100", "count": "8", "training_time": "12h"},
            "evaluation": {"metrics": ["accuracy"], "protocol": "test split", "few_shot": "5-shot"},
            "ablations": [{"setting": "no acting", "finding": "-3 acc"}],
            "reproducibility": {
                "code_available": "true", "code_url": "https://github.com/a/b",
                "key_to_reproduce": ["prompt format"], "missing_details": ["seed"],
            },
        }
        card = utils.build_setup_result_card(setup)
        blob = json.dumps(card, ensure_ascii=False)
        self.assertIn("实验配置", blob)
        self.assertIn("GSM8K", blob)
        self.assertIn("A100", blob)
        self.assertIn("5-shot", blob)
        self.assertIn("https://github.com/a/b", blob)

    @patch("extract_setup.llm_chat")
    def test_extract_setup_feeds_experiment_evidence_and_fills_ids(self, chat):
        import extract_setup

        chat.return_value = '{"title":"","model":{"name":"PaLM"}}'
        paper = {
            "title": "ReAct", "arxiv_id": "2210.03629",
            "abstract_url": "https://arxiv.org/abs/2210.03629", "summary": "abstract text",
        }
        evidence = {"first_page": "fp", "experiment": "EXPERIMENT SECTION TEXT", "code_urls": ["https://github.com/a/b"]}

        out = extract_setup.extract_setup(paper, ocr_evidence=evidence, model="test")

        self.assertEqual(out["arxiv_id"], "2210.03629")
        self.assertEqual(out["title"], "ReAct")  # empty title backfilled from paper
        user_msg = chat.call_args.kwargs["messages"][1]["content"]
        self.assertIn("EXPERIMENT SECTION TEXT", user_msg)  # OCR 实验证据被喂给模型


def _make_zip_bytes(markdown: str) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("full.md", markdown)
        zf.writestr("layout.json", "{}")
    return buffer.getvalue()


class OCRTransportTests(unittest.TestCase):
    def test_normalize_accepts_abs_pdf_and_raw_id(self):
        self.assertEqual(
            utils._normalize_arxiv_pdf_url("https://arxiv.org/abs/1234.5678"),
            ("1234.5678", "https://arxiv.org/pdf/1234.5678"),
        )
        self.assertEqual(
            utils._normalize_arxiv_pdf_url("https://arxiv.org/pdf/1234.5678.pdf"),
            ("1234.5678", "https://arxiv.org/pdf/1234.5678"),
        )
        self.assertEqual(
            utils._normalize_arxiv_pdf_url("1234.5678"),
            ("1234.5678", "https://arxiv.org/pdf/1234.5678"),
        )

    @patch.object(utils, "MINERU_TOKEN", "")
    @patch("utils.time.sleep")
    @patch("utils.httpx.post")
    @patch("utils.httpx.get")
    def test_agent_path_submits_pdf_url_without_local_download(self, get, post, _sleep):
        submitted = Mock()
        submitted.status_code = 200
        submitted.json.return_value = {"code": 0, "data": {"task_id": "task-1"}}
        post.return_value = submitted

        poll = Mock()
        poll.status_code = 200
        poll.json.return_value = {"code": 0, "data": {"state": "done", "markdown_url": "https://cdn/full.md"}}
        markdown = Mock()
        markdown.text = "# Title\nbody"
        markdown.raise_for_status.return_value = None
        get.side_effect = [poll, markdown]

        with tempfile.TemporaryDirectory() as tmp:
            result = utils.ocr_arxiv_pdf("https://arxiv.org/abs/1234.5678", output_dir=tmp)

        # 无 Token → 走 Agent 轻量解析；accept abs，归一化为 pdf 链接交给服务端
        self.assertEqual(result["markdown"], "# Title\nbody")
        self.assertEqual(result["pdf_url"], "https://arxiv.org/pdf/1234.5678")
        self.assertEqual(result["ocr_source"], "agent")

        kwargs = post.call_args.kwargs
        self.assertNotIn("files", kwargs)  # 不再 multipart 上传本地文件
        self.assertEqual(kwargs["json"]["url"], "https://arxiv.org/pdf/1234.5678")
        self.assertEqual(kwargs["json"]["page_range"], "1-20")

        # 不向 arxiv.org 发起任何下载请求（服务端负责下载 PDF）
        for call in get.call_args_list:
            self.assertNotIn("arxiv.org", call.args[0])

    @patch.object(utils, "MINERU_TOKEN", "tok")
    @patch("utils.time.sleep")
    @patch("utils.httpx.post")
    @patch("utils.httpx.get")
    def test_precision_path_preferred_when_token_set(self, get, post, _sleep):
        submitted = Mock()
        submitted.status_code = 200
        submitted.json.return_value = {"code": 0, "data": {"task_id": "task-1"}}
        post.return_value = submitted

        poll = Mock()
        poll.status_code = 200
        poll.json.return_value = {"code": 0, "data": {"state": "done", "full_zip_url": "https://cdn/result.zip"}}
        zip_resp = Mock()
        zip_resp.content = _make_zip_bytes("# Precise\nbody")
        zip_resp.raise_for_status.return_value = None
        get.side_effect = [poll, zip_resp]

        with tempfile.TemporaryDirectory() as tmp:
            result = utils.ocr_arxiv_pdf("https://arxiv.org/abs/1234.5678", output_dir=tmp)

        self.assertEqual(result["markdown"], "# Precise\nbody")
        self.assertEqual(result["ocr_source"], "precision")

        kwargs = post.call_args.kwargs
        self.assertEqual(post.call_args.args[0], utils.MINERU_PRECISION_URL)
        self.assertIn("Bearer tok", kwargs["headers"]["Authorization"])
        self.assertEqual(kwargs["json"]["page_ranges"], "1-20")
        self.assertEqual(kwargs["json"]["model_version"], "vlm")

        for call in get.call_args_list:
            self.assertNotIn("arxiv.org", call.args[0])

    @patch.object(utils, "MINERU_TOKEN", "tok")
    @patch("utils.time.sleep")
    @patch("utils.httpx.post")
    @patch("utils.httpx.get")
    def test_precision_failure_falls_back_to_agent(self, get, post, _sleep):
        precision_reject = Mock()
        precision_reject.status_code = 200
        precision_reject.json.return_value = {"code": -1, "msg": "quota exceeded"}
        agent_ok = Mock()
        agent_ok.status_code = 200
        agent_ok.json.return_value = {"code": 0, "data": {"task_id": "agent-1"}}
        post.side_effect = [precision_reject, agent_ok]

        poll = Mock()
        poll.status_code = 200
        poll.json.return_value = {"code": 0, "data": {"state": "done", "markdown_url": "https://cdn/full.md"}}
        markdown = Mock()
        markdown.text = "# Fallback\nbody"
        markdown.raise_for_status.return_value = None
        get.side_effect = [poll, markdown]

        with tempfile.TemporaryDirectory() as tmp:
            result = utils.ocr_arxiv_pdf("https://arxiv.org/abs/1234.5678", output_dir=tmp)

        self.assertEqual(result["markdown"], "# Fallback\nbody")
        self.assertEqual(result["ocr_source"], "agent")
        self.assertEqual(post.call_args_list[1].args[0], utils.MINERU_AGENT_URL)

    @patch.object(utils, "MINERU_TOKEN", "")
    @patch("utils.time.sleep")
    @patch("utils.httpx.post")
    @patch("utils.httpx.get")
    def test_ocr_returns_none_on_failed_state(self, get, post, _sleep):
        submitted = Mock()
        submitted.status_code = 200
        submitted.json.return_value = {"code": 0, "data": {"task_id": "task-1"}}
        post.return_value = submitted

        poll = Mock()
        poll.status_code = 200
        poll.json.return_value = {"code": 0, "data": {"state": "failed", "err_msg": "unsupported"}}
        get.return_value = poll

        self.assertIsNone(utils.ocr_arxiv_pdf("https://arxiv.org/abs/1234.5678"))


class ArxivResilienceTests(unittest.TestCase):
    def test_arxiv_user_agent_format(self):
        ua = utils.get_arxiv_user_agent()
        self.assertIn("ArxivDigest", ua)

    @patch("utils.time.sleep")
    @patch("utils.urllib.request.urlopen")
    def test_robust_arxiv_request_retries_on_429(self, mock_urlopen, mock_sleep):
        err_429 = urllib.error.HTTPError(
            url="https://export.arxiv.org",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "2"},
            fp=None,
        )
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = b"<xml>success</xml>"
        mock_urlopen.side_effect = [err_429, mock_resp]

        content = utils.robust_arxiv_request("https://export.arxiv.org", max_retries=2, base_delay=0.1)
        self.assertEqual(content, "<xml>success</xml>")
        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called()

    @patch("utils.fetch_arxiv_papers_rss")
    @patch("utils.robust_arxiv_request")
    def test_fetch_arxiv_papers_falls_back_to_rss_on_failure(self, mock_req, mock_rss):
        mock_req.side_effect = RuntimeError("429 Throttled")
        mock_rss.return_value = [{"title": "Fallback Paper", "arxiv_id": "2609.00001"}]

        papers = utils.fetch_arxiv_papers(categories=["cs.AI"], keywords=[], lookback_days=1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "Fallback Paper")
        mock_rss.assert_called_once()

    @patch("utils.robust_arxiv_request")
    def test_fetch_arxiv_papers_rss_parses_feed(self, mock_req):
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
            <channel>
                <title>cs.AI</title>
                <item>
                    <title>RSS Test Paper</title>
                    <link>https://arxiv.org/abs/2609.12345</link>
                    <description>arXiv:2609.12345v1 Announce Type: new &#10;Abstract: This is a test abstract.</description>
                    <dc:creator>Alice Smith, Bob Jones</dc:creator>
                    <pubDate>Sat, 05 Sep 2026 00:00:00 -0400</pubDate>
                </item>
            </channel>
        </rss>
        """
        mock_req.return_value = rss_xml
        papers = utils.fetch_arxiv_papers_rss(["cs.AI"], lookback_days=7)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "RSS Test Paper")
        self.assertEqual(papers[0]["arxiv_id"], "2609.12345")
        self.assertEqual(papers[0]["summary"], "This is a test abstract.")
        self.assertEqual(papers[0]["authors"], ["Alice Smith", "Bob Jones"])
        self.assertEqual(papers[0]["abstract_url"], "https://arxiv.org/abs/2609.12345")
        self.assertEqual(papers[0]["pdf_url"], "https://arxiv.org/pdf/2609.12345")

    @patch("reading.fetch_single_paper_from_abs")
    @patch("reading.robust_arxiv_request")
    def test_fetch_single_paper_falls_back_to_abs_on_error(self, mock_req, mock_abs):
        import reading
        mock_req.side_effect = RuntimeError("429 Too Many Requests")
        mock_abs.return_value = {
            "title": "Abs Fallback Title",
            "summary": "Abs summary",
            "arxiv_id": "2609.12345",
            "abstract_url": "https://arxiv.org/abs/2609.12345",
            "pdf_url": "https://arxiv.org/pdf/2609.12345",
            "authors": ["Author One"],
        }
        paper = reading.fetch_single_paper("2609.12345")
        self.assertEqual(paper["title"], "Abs Fallback Title")
        mock_abs.assert_called_once_with("2609.12345")

    @patch("reading.robust_arxiv_request")
    def test_fetch_single_paper_from_abs_parses_html(self, mock_req):
        import reading
        html = """
        <html>
        <head>
            <meta name="citation_title" content="Deep Learning Breakthrough" />
            <meta name="citation_author" content="LeCun, Yann" />
            <meta name="citation_author" content="Bengio, Yoshua" />
            <meta name="citation_date" content="2026/09/01" />
        </head>
        <body>
            <blockquote class="abstract mathjax">
                <span class="descriptor">Abstract:</span> We present a new method.
            </blockquote>
        </body>
        </html>
        """
        mock_req.return_value = html
        paper = reading.fetch_single_paper_from_abs("2609.99999")
        self.assertEqual(paper["title"], "Deep Learning Breakthrough")
        self.assertEqual(paper["authors"], ["Yann LeCun", "Yoshua Bengio"])
        self.assertEqual(paper["summary"], "We present a new method.")
        self.assertEqual(paper["arxiv_id"], "2609.99999")


if __name__ == "__main__":
    unittest.main()
