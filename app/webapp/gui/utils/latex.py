from nicegui import ui
import re

# === Load MathJax once ===
if not hasattr(ui, '_mathjax_loaded'):
    ui.add_body_html(r"""
    <script>
    window.mathjax_loaded = false;
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      },
      svg: { fontCache: 'global' }
    };
    </script>
    <script>
    window.mathjax_loaded = true;
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    """)
    ui._mathjax_loaded = True


def typeset_latex():
    """Schedule MathJax typesetting with a short delay"""
    def try_typeset():
        ui.run_javascript("""
        if (window.MathJax && window.mathjax_loaded) {
            MathJax.typeset();
        } else {
            setTimeout(() => MathJax.typeset(), 50);
        }
        """)
    ui.timer(0.2, try_typeset, once=True)


class LatexLabel:
    def __init__(self, text: str):
        self.text = text
        self.element = self._render(text)
        typeset_latex()

    def _contains_latex(self, text: str) -> bool:
        return (
            re.search(r'\$\$(.*?)\$\$', text, re.DOTALL) or
            r'\begin{' in text
        )

    def _render(self, text: str):
        if self._contains_latex(text):
            html_element = ui.html(f'<p>{text}</p>')
            return html_element
        else:
            return ui.markdown(text)

    def update(self, new_text: str):
        self.text = new_text
        self.element.clear()
        self.element = self._render(new_text)
        typeset_latex()
