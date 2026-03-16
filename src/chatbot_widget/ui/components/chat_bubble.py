import ipywidgets as widgets
from markdown import markdown
from markdown.extensions.codehilite import CodeHiliteExtension

class ChatBubble:
    """Single chat message bubble with subtle gradient and animation."""

    def __init__(self, text: str, sender: str = "bot"):
        if sender == "user":
            self._bg = "linear-gradient(135deg, #fff4e5 0%, #ffe1b3 100%)"
            self._align = "flex-end"
            self._border_radius = "18px 18px 4px 18px"
            self._text_color = "#3a2f00"
        else:
            self._bg = "linear-gradient(135deg, #f6f8fa 0%, #eaeef3 100%)"
            self._align = "flex-start"
            self._border_radius = "18px 18px 18px 4px"
            self._text_color = "#1a1a1a"

        self.widget = widgets.HTML(value=self._build_html(text))

    def _build_html(self, text) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        try:
            rendered = markdown(
                text,
                extensions=["fenced_code", "tables", CodeHiliteExtension(noclasses=True, pygments_style="default")]
            )
        except Exception:
            rendered = f"<pre>{text}</pre>"
        return self._wrap_html(rendered)

    def _wrap_html(self, rendered: str) -> str:
        return f"""
            <style>
            @keyframes cb-fade-in {{
                from {{ opacity: 0; transform: translateY(4px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            </style>
            <div style='display:flex;justify-content:{self._align};
                        margin:8px 0;
                        animation:cb-fade-in 0.2s ease-out;'>
              <div style='
                  background:{self._bg};
                  color:{self._text_color};
                  padding:0px 14px;
                  border-radius:{self._border_radius};
                  max-width:90%;
                  overflow-x:auto;
                  box-shadow:0 4px 10px rgba(0,0,0,0.08);
                  font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;
                  font-size:15px;
                  line-height:1.2;
                  margin: 0;
              '>
                {rendered}
              </div>
            </div>
            """

    def update_text(self, text: str):
        """Update the bubble content in-place (used for streaming)."""
        self.widget.value = self._build_html(text)
