import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import utils
import digest
import paper_context


class WorkflowContractTests(unittest.TestCase):
    def test_paper_analysis_has_issue_permission_and_enables_ocr(self):
        workflow = (ROOT / ".github/workflows/paper-analysis.yml").read_text(encoding="utf-8")

        self.assertIn("issues: write", workflow)
        self.assertIn("OCR_API_TOKEN: ${{ secrets.OCR_API_TOKEN }}", workflow)
        self.assertIn("--use-ocr --push-to-feishu", workflow)
        self.assertNotIn("FEISHU_WEBHOOK: ${{ secrets.FEISHU_WEBHOOK }}", workflow)
        self.assertIn("d.get('quick_take'", workflow)
        self.assertNotIn("d.get('core_claim'", workflow)

    def test_daily_digest_enables_ocr_and_uploads_artifact(self):
        workflow = (ROOT / ".github/workflows/daily-digest.yml").read_text(encoding="utf-8")
        self.assertIn("OCR_API_TOKEN: ${{ secrets.OCR_API_TOKEN }}", workflow)
        self.assertIn("output/digest_analysis.json", workflow)

    def test_cloudflare_worker_was_removed(self):
        self.assertFalse((ROOT / "cloudflare-worker/index.js").exists())


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


class OCRTransportTests(unittest.TestCase):
    @patch("utils.time.sleep")
    @patch("utils.httpx.post")
    @patch("utils.httpx.get")
    def test_ocr_uploads_a_valid_downloaded_pdf_as_multipart(self, get, post, _sleep):
        pdf = unittest.mock.Mock()
        pdf.content = b"%PDF-1.7 fixture"
        pdf.headers = {"content-type": "application/pdf"}
        pdf.raise_for_status.return_value = None

        status = unittest.mock.Mock()
        status.status_code = 200
        status.json.return_value = {"data": {"state": "failed", "errorMsg": "stop fixture"}}
        get.side_effect = [pdf, status]

        submitted = unittest.mock.Mock()
        submitted.status_code = 200
        submitted.json.return_value = {"data": {"jobId": "job-1"}}
        post.return_value = submitted

        with patch.object(utils, "OCR_API_TOKEN", "token"):
            self.assertIsNone(utils.ocr_arxiv_pdf("https://arxiv.org/abs/1234.5678"))

        kwargs = post.call_args.kwargs
        self.assertIn("files", kwargs)
        self.assertTrue(kwargs["files"]["file"][1].startswith(b"%PDF"))
        self.assertNotIn("fileUrl", kwargs["data"])


if __name__ == "__main__":
    unittest.main()
