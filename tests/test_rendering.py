from __future__ import annotations

import unittest
from pathlib import Path

from local_rag.rendering import SafeMarkdownRenderer


class RenderingTests(unittest.TestCase):
    def test_markdown_table_is_preserved_and_script_is_removed(self) -> None:
        rendered = SafeMarkdownRenderer().render(
            "| 欄位 | 值 |\n| --- | --- |\n| A | B |\n<script>alert(1)</script>"
        )
        self.assertIn("<table>", rendered)
        self.assertIn("<th>欄位</th>", rendered)
        self.assertNotIn("<script", rendered)

    def test_frontend_contains_citation_cards_and_lightbox(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "static" / "app.js").read_text(encoding="utf-8")
        styles = (root / "static" / "app.css").read_text(encoding="utf-8")
        self.assertIn("citation-card", script)
        self.assertIn("citation-lightbox", script)
        self.assertIn(".citation-grid", styles)
        self.assertIn(".citation-lightbox", styles)


if __name__ == "__main__":
    unittest.main()
