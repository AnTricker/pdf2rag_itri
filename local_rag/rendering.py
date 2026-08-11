from __future__ import annotations

import html


class SafeMarkdownRenderer:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def render(self, text: str) -> str:
        if not self.enabled:
            return f"<p>{html.escape(text)}</p>"
        import bleach
        import markdown

        rendered = markdown.markdown(
            text,
            extensions=["fenced_code", "sane_lists"],
            output_format="html",
        )
        return bleach.clean(
            rendered,
            tags={
                "p",
                "br",
                "strong",
                "em",
                "h1",
                "h2",
                "h3",
                "ul",
                "ol",
                "li",
                "blockquote",
                "pre",
                "code",
                "a",
            },
            attributes={"a": ["href", "title"]},
            protocols={"http", "https"},
            strip=True,
        )
