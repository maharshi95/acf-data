# %%
import importlib
import re
from typing import NamedTuple

from IPython.display import HTML, display

import models
from utils import acf_sanitization

acf_sanitization = importlib.reload(acf_sanitization)


session = models.create_session("data/acf-23-24.db")
tossups = session.query(models.Tossup).all()


TossupEntry = NamedTuple(
    "TossupEntry",
    [
        ("id", str),
        ("question_raw", str),
        ("question", str),
    ],
)

tossup_entries = []
for t in tossups:
    tossup_entries.append(
        TossupEntry(
            id=t.id,
            question_raw=t.question_text,
            question=acf_sanitization.sanitize_question(t.question_text),
        )
    )


def search_and_highlight_pattern(pattern: str, raw: bool = False, limit: int = 10):
    html_outputs = []
    for t in tossup_entries:
        question = t.question_raw if raw else t.question
        if re.search(pattern, question):
            html_output = f"""
            <div style="background-color: #f0f8ff; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">Tossup ID: <span style="background-color: #ffd700; padding: 2px 5px; border-radius: 5px;">{t.id}</span></h3>
                <div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; font-size: 16px; line-height: 1.6;">
            """
            highlighted_question = re.sub(
                pattern,
                lambda m: f'<span style="background-color: #ff4500; color: white; padding: 2px 4px; border-radius: 4px; font-weight: bold;">{m.group()}</span>',
                question,
            )
            html_output += f"{highlighted_question}</div></div>"
            html_outputs.append(html_output)

    total_outputs = len(html_outputs)
    print(f"Total matches found: {total_outputs}")

    html_output = f"""
    <h2>Searching for <code style="background-color: #e0e0e0; padding: 2px 4px; border-radius: 4px;">{pattern}</code> in <code style="background-color: #e0e0e0; padding: 2px 4px; border-radius: 4px;">{'raw' if raw else 'sanitized'}</code> tossup questions:</h3>
    """

    html_output += "\n".join(html_outputs[:limit])
    if total_outputs > limit:
        html_output += f"""
        <div style="text-align: center; margin-top: 20px; font-style: italic; color: #666;">
            Showing {limit} out of {total_outputs} total matches.
        </div>
        """

    display(HTML(html_output))


# Example usage:
# Search for quotes containing semicolons in raw questions
pattern = r"\[.*?\]"
search_and_highlight_pattern(pattern, raw=False)

# Search for semicolons in sanitized questions
# search_and_highlight_pattern(r";", raw=False)

# %%
