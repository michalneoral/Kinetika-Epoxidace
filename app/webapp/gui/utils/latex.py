# app/webapp/gui/utils/latex.py

from nicegui import ui
import re

# === Load MathJax once ===
if not hasattr(ui, '_mathjax_loaded'):
    ui.add_body_html(r"""
    <script>
      // Configure MathJax before loading the script
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\(', '\\)']],
          displayMath: [['$$', '$$'], ['\\[', '\\]']]
        },
        svg: { fontCache: 'global' }
      };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    """)
    ui._mathjax_loaded = True


def typeset_latex():
    """Ask MathJax to typeset the page (queued by NiceGUI until page is ready)."""
    ui.run_javascript("""
        if (window.MathJax) {
            MathJax.typeset();
        }
    """)


class LatexLabel:
    def __init__(self, text: str):
        self.text = text
        self.element = self._render(text)
        typeset_latex()
        raise NotImplementedError('This part of code does not work after update')

    def _contains_latex(self, text: str) -> bool:
        return (
            re.search(r'\$\$(.*?)\$\$', text, re.DOTALL)
            or r'\\begin{' in text
        )

    def _render(self, text: str):
        if self._contains_latex(text):
            # IMPORTANT in NiceGUI 3: disable sanitization for raw LaTeX markup
            return ui.html(f'<p>{text}</p>', sanitize=False)
        else:
            return ui.markdown(text)

    def update(self, new_text: str):
        self.text = new_text
        self.element.clear()
        self.element = self._render(new_text)
        typeset_latex()
