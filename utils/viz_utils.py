import tempfile
import webbrowser
from typing import Any, Dict


def display_interactive_tokens(tokens):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive Tokens</title>
        <style>
            .token {{
                display: inline-block;
                margin: 2px;
                padding: 2px 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                cursor: pointer;
            }}
            #index-display {{
                margin-top: 20px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div id="tokens-container">
            {tokens_html}
        </div>
        <div id="index-display"></div>
        <script>
            const tokens = document.querySelectorAll('.token');
            const indexDisplay = document.getElementById('index-display');
            
            tokens.forEach((token, index) => {{
                token.addEventListener('mouseover', () => {{
                    indexDisplay.textContent = `Token Index: ${{index}}`;
                }});
                token.addEventListener('mouseout', () => {{
                    indexDisplay.textContent = '';
                }});
            }});
        </script>
    </body>
    </html>
    """

    tokens_html = "".join([f'<span class="token">{token}</span>' for token in tokens])
    final_html = html_content.format(tokens_html=tokens_html)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
        f.write(final_html)
    webbrowser.open("file://" + f.name)


class DiffVisualizer:
    def __init__(self, diff_dict: Dict[str, Dict[str, Any]]):
        self.diff_dict = diff_dict

    def __str__(self) -> str:
        return self.format_diff()

    def format_diff(self) -> str:
        formatted_diff = []
        for key, values in self.diff_dict.items():
            formatted_diff.append(f"\t{key}: {values['old']} -> {values['new']}")
        return "Differences:\n" + "\n".join(formatted_diff)
