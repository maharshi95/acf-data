# %%
import importlib
import textwrap
from collections import Counter

import numpy as np
from IPython.display import HTML, display
from matplotlib import pyplot as plt
from nltk.tokenize import PunktSentenceTokenizer
from rich import print as rprint

import models
from structs import create_tossup_entry
from utils import acf_sanitization, qb_tokenization

qb_tokenization = importlib.reload(qb_tokenization)
acf_sanitization = importlib.reload(acf_sanitization)


# %%
session = models.create_session("data/acf-23-24.db")
tossups = session.query(models.Tossup).all()
tossups_by_id = {t.id: t for t in tossups}

punkt_sent_tokenizer = PunktSentenceTokenizer()


ques = [
    'Two musicians’ parts start one bar apart and gradually merge in a canon from this piece that starts with the quarter notes "B, C-sharp, A, G-sharp, F-sharp, E." Reminders to play molto dolce, sempre dolce, and dolcissimo intensify under a rocking motif in this piece that is first stated as a broken dominant  ninth chord and is developed in this piece’s Recitativo-Fantasia.',
    'I was walking by "the strees." In the novel "The Secret History" by Donna Tartt, the narrator walks on "The 1st Street." by the road, only to be greeted by "the 1st Street."',
    "The clue is “How about you.” Second clue goes, “are you doing? I am fine, thank you. How about you?”",
    "“How.” he said, “are you doing? I am fine, thank you. How about you?”",
    "“Torched it” was the same part of this larger body of work that contained “Are you man.” \"Toil and Trouble\" wasn't though.However, the clue giver wasn't quite sure of it.",
    "J. F. Kennedy, but not by these other two researchers.",
    "These institutions are using the Bath Protocol to remedy the outdated Z39.50 protocol. S. R. Ranganathan’s fifth rule of these institutions calls them “growing organisms.”",
    'Charles Beem "revisited" this event in the chapter "What Power Have I Left?" in The Lioness Roared.',
    'In a novel titled for one of these periods, the family of a "lanie history teacher is scandalized by a photo of him hugging the widow of his school’s janitor. In that novel titled one of these periods, the Special Branch’s murder of Gordon Ngubene is investigated by the Afrikaner Ben du Toit. For 10 points, André Brink titled a novel for a "dry white" sort of what time period?',
]

print("## Tests")
for q in ques:
    q = q.replace("“", '"').replace("”", '"')
    print("Test Question:")
    print(textwrap.indent(textwrap.fill(q), "    "))
    spans = qb_tokenization.get_clue_spans(q, tokenization_scheme="punkt")
    if not spans:
        print("No clue spans found")
    else:
        print("Clue Spans:")
        for i, s in enumerate(spans):
            print(f"{i+1}: {q[s[0] : s[1]]}")
    print()

# %%

tossup_entries = [create_tossup_entry(t) for t in tossups]


clues = []
for i, t in enumerate(tossup_entries):
    for c in t.clues():
        clues.append((c, i))
clues_sorted = sorted(clues, key=lambda x: len(x[0].split()), reverse=True)

# Distributions of # tokens per clue
n_tokens_per_clue = [len(c.split()) for c, i in clues_sorted]
plt.figure(figsize=(10, 6))
plt.hist(n_tokens_per_clue, bins=49, color="skyblue", edgecolor="black")
plt.title("Distribution of Tokens per Clue", fontsize=16)
plt.xlabel("Number of Tokens", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
# %%
# Distributions of n_clues
n_clues = [len(t.clues()) for t in tossup_entries]
min_clue_count = min(n_clues)
max_clue_count = max(n_clues)
plt.figure(figsize=(10, 6))
plt.bar(
    range(min_clue_count, max_clue_count + 1),
    [n_clues.count(i) for i in range(min_clue_count, max_clue_count + 1)],
    color="lightpink",
    edgecolor="black",
)
plt.title("Distribution of Number of Clues per Tossup", fontsize=16)
plt.xlabel("Number of Clues", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(range(min_clue_count, max_clue_count + 1))
plt.grid(True, linestyle="--", alpha=0.7, axis="y")
plt.tight_layout()
plt.show()

# %%

html_output = ""
for c, i in clues_sorted[-20:]:
    html_output += f"""
    <div style="background-color: #f0f8ff; border-radius: 15px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h2 style="color: #2c3e50;">Tossup ID: {tossup_entries[i].qid}</h2>
        <h2 style="color: #1e90ff; text-align: center; text-transform: uppercase; letter-spacing: 2px;">Shortest Clue Analysis</h2>
        <div style="background-color: #fffaf0; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #ff4500; margin-bottom: 10px;">Shortest Clue:</h3>
            <p style="font-size: 18px; font-weight: bold; color: #2f4f4f;">{c}</p>
        </div>
        <div style="background-color: #f0fff0; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #228b22; margin-bottom: 10px;">Full Question:</h3>
            <p style="font-size: 16px; color: #2f4f4f; line-height: 1.6;">
                {textwrap.fill(tossup_entries[i].question, width=80)}
            </p>
        </div>
        <div style="background-color: #fff0f5; padding: 15px; border-radius: 10px;">
            <h3 style="color: #8b008b; margin-bottom: 10px;">All Clues:</h3>
            <ol style="color: #4b0082; font-size: 16px;">
    """
    all_clues = tossup_entries[i].clues()
    shortest_clue = min(all_clues, key=len)
    for j, cl in enumerate(all_clues):
        if cl == shortest_clue:
            html_output += f"<li style='margin-bottom: 10px; background-color: #c0cccc; padding: 5px; border-radius: 5px;'>{cl}</li>"
        else:
            html_output += f"<li style='margin-bottom: 10px;'>{cl}</li>"
    html_output += """
            </ol>
        </div>
    </div>
    """

display(HTML(html_output))
# %%


html_output = ""
for c, i in clues_sorted[:20]:
    question_text = tossup_entries[i].question
    html_output += f"""
    <div style="background-color: #f0f0f0; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
        <h3 style="color: #2c3e50;">Tossup ID: {tossup_entries[i].qid}</h3>
        <p style="font-size: 20px; color: #e74c3c;"><strong>Longest clue:</strong> {len(c.split())} words</p>
        <div style="background-color: #ecf0f1; padding: 10px; border-left: 5px solid #3498db; margin-bottom: 10px;">
            <p style="font-style: italic; font-size: 18px;">{c}</p>
        </div>
        <h4 style="color: #2c3e50;">Full Question:</h4>
        <div style="background-color: #ecf0f1; padding: 10px; border-left: 5px solid #2ecc71; margin-bottom: 10px;">
            <p style="font-size: 16px;">{question_text}</p>
        </div>
        <h4 style="color: #2c3e50;">All Clues:</h4>
        <ol style="color: #34495e; font-size: 16px;">
    """
    clues = tossup_entries[i].clues()
    longest_clue = max(clues, key=len)
    for j, cl in enumerate(clues):
        if cl == longest_clue:
            html_output += (
                f'<li style="background-color: #fffacd; padding: 5px;">{cl}</li>'
            )
        else:
            html_output += f"<li>{cl}</li>"
    html_output += """
        </ol>
    </div>
    """
    # Add Punkt tokenizer split for the longest clue
    punkt_spans = qb_tokenization.punkt_sent_tokenizer.span_tokenize(longest_clue)
    punkt_clues = [longest_clue[s:e] for s, e in punkt_spans]
    html_output += """
        <h4 style="color: #2c3e50;">Punkt Tokenizer Split (Longest Clue):</h4>
        <ol style="color: #34495e; font-size: 16px;">
    """
    for punkt_clue in punkt_clues:
        html_output += f"<li>{punkt_clue}</li>"
    html_output += """
        </ol>
    """

display(HTML(html_output))
# %%
from utils.qb_tokenization import generate_blingfire_spans, generate_punkt_sent_spans

for t in tossups:
    if t.id == 1843:
        print(textwrap.fill(t.question_text))
        q_sanitized = acf_sanitization.sanitize_question(t.question_text)
        break


bf_spans = list(generate_blingfire_spans(q_sanitized))
bf_sents = [q_sanitized[s[0] : s[1]] for s in bf_spans]
rprint(bf_sents)
punkt_spans = list(generate_punkt_sent_spans(q_sanitized))
punkt_sents = [q_sanitized[s[0] : s[1]] for s in punkt_spans]
rprint(punkt_sents)


# %%
qb_tokenization = importlib.reload(qb_tokenization)
# Difference in blingfire and punkt tokenization
diff_list = []
all_bf_clues = []
all_punkt_clues = []
bf_n_clues = []
punkt_n_clues = []
for t in tossups:
    question_text = acf_sanitization.sanitize_question(t.question_text)
    question_text = question_text.replace("“", '"').replace("”", '"')
    try:
        bf_spans = qb_tokenization.generate_blingfire_spans(question_text)
        punkt_spans = qb_tokenization.generate_punkt_sent_spans(question_text)
        all_bf_clues.extend([question_text[s[0] : s[1]] for s in bf_spans])
        all_punkt_clues.extend([question_text[s[0] : s[1]] for s in punkt_spans])
        bf_n_clues.append(len(bf_spans))
        punkt_n_clues.append(len(punkt_spans))
        assert len(bf_spans) > 0, f"No blingfire spans found for question: {t.id}"
        assert (
            len(punkt_spans) > 0
        ), f"No punkt spans found for tossup: {t.id} \nQuestion: {question_text}"
    except RuntimeError as e:
        print("Error while tokenizing tossup ", t.id)
        print(textwrap.fill(question_text))
        raise e
    if bf_spans != punkt_spans:
        diff_list.append(
            {
                "id": t.id,
                "question": question_text,
                "bf_spans": bf_spans,
                "punkt_spans": punkt_spans,
            }
        )

print("# differences: ", len(diff_list), "out of ", len(tossups))

# Sort examples by the difference in number of spans
examples = sorted(
    diff_list,
    key=lambda x: abs(len(x["bf_spans"]) - len(x["punkt_spans"])),
    reverse=False,
)[:10]

# Create HTML visualization
html_output = """
<style>
    .tokenization-example {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .tokenization-header {
        color: #2c3e50;
        font-size: 18px;
    }
    .tokenization-text {
        background-color: #ecf0f1;
        padding: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 10px;
    }
    .tokenization-comparison {
        display: flex;
        justify-content: space-between;
    }
    .tokenization-spans {
        background-color: #e8f6f3;
        padding: 10px;
        border-left: 5px solid #1abc9c;
        width: 48%;
    }
    .span-item {
        margin-bottom: 5px;
    }
</style>
"""

for example in examples:
    bf_clues = set(example["question"][s:e] for s, e in example["bf_spans"])
    punkt_clues = set(example["question"][s:e] for s, e in example["punkt_spans"])

    different_bf_spans = [
        (s, e)
        for s, e in example["bf_spans"]
        if example["question"][s:e] not in punkt_clues
    ]
    different_punkt_spans = [
        (s, e)
        for s, e in example["punkt_spans"]
        if example["question"][s:e] not in bf_clues
    ]

    html_output += f"""
    <div class="tokenization-example">
        <h3 class="tokenization-header">Tossup ID: {example['id']}</h3>
        <div class="tokenization-text">
            <p>{example['question']}</p>
        </div>
        <div class="tokenization-comparison">
            <div class="tokenization-spans">
                <h4 class="tokenization-header"># BlingFire Spans: {len(example["bf_spans"])}</h4>
                {''.join(f'<p class="span-item">{example["question"][s:e]}</p>' for s, e in different_bf_spans)}
            </div>
            <div class="tokenization-spans">
                <h4 class="tokenization-header"># Punkt Spans: {len(example["punkt_spans"])}</h4>
                {''.join(f'<p class="span-item">{example["question"][s:e]}</p>' for s, e in different_punkt_spans)}
            </div>
        </div>
    </div>
    """

display(HTML(html_output))


# %%
# Prepare data for plotting
bf_clue_lengths = [len(clue.split()) for clue in all_bf_clues]
punkt_clue_lengths = [len(clue.split()) for clue in all_punkt_clues]

# Set up the plots
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, 1]}
)

# Plot histograms
ax1.hist(
    bf_clue_lengths,
    bins=50,
    alpha=0.7,
    label="BlingFire",
    color="skyblue",
    edgecolor="black",
)
ax1.hist(
    punkt_clue_lengths,
    bins=50,
    alpha=0.7,
    label="Punkt",
    color="lightgreen",
    edgecolor="black",
)

# Customize the histogram plot
ax1.set_title("Distribution of Clue Lengths: BlingFire vs Punkt", fontsize=16)
ax1.set_xlabel("Number of Tokens per Clue", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.7)

# Add some statistics to the histogram plot
bf_mean = np.mean(bf_clue_lengths)
punkt_mean = np.mean(punkt_clue_lengths)
ax1.axvline(
    bf_mean,
    color="blue",
    linestyle="dashed",
    linewidth=2,
    label=f"BlingFire Mean: {bf_mean:.2f}",
)
ax1.axvline(
    punkt_mean,
    color="green",
    linestyle="dashed",
    linewidth=2,
    label=f"Punkt Mean: {punkt_mean:.2f}",
)

ax1.legend(fontsize=10)

# Prepare data for the vertical barplot
bf_clue_counts = Counter(bf_n_clues)
punkt_clue_counts = Counter(punkt_n_clues)

min_clues = min(min(bf_clue_counts.keys()), min(punkt_clue_counts.keys()))
max_clues = max(max(bf_clue_counts.keys()), max(punkt_clue_counts.keys()))
clue_range = range(min_clues, max_clues + 1)

bf_counts = [bf_clue_counts.get(i, 0) for i in clue_range]
punkt_counts = [punkt_clue_counts.get(i, 0) for i in clue_range]

# Plot vertical barplot
x = np.arange(len(clue_range))
width = 0.35

ax2.bar(
    x - width / 2,
    bf_counts,
    width,
    label="BlingFire",
    color="skyblue",
    edgecolor="black",
)
ax2.bar(
    x + width / 2,
    punkt_counts,
    width,
    label="Punkt",
    color="lightgreen",
    edgecolor="black",
)

# Customize the barplot
ax2.set_title("Distribution of Number of Clues: BlingFire vs Punkt", fontsize=16)
ax2.set_xlabel("Number of Clues", fontsize=12)
ax2.set_ylabel("Frequency", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(clue_range)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.7, axis="y")

# Show the plots
plt.tight_layout()
plt.show()

# Print some additional statistics
print(f"BlingFire - Mean: {bf_mean:.2f}, Median: {np.median(bf_clue_lengths):.2f}")
print(f"Punkt - Mean: {punkt_mean:.2f}, Median: {np.median(punkt_clue_lengths):.2f}")


# %%
text = """After tackling an ambassador taking pictures of
“the Big Board” here, a line about this location is delivered by
President Merkin Muffley. After rising from his wheelchair in this
location, a Nazi scientist yells “Mein Führer, I can walk!” For 10
points, a line beginning “Gentlemen! You can’t fight in here” refers
to what location in the Pentagon, the setting of a political satire by
Stanley Kubrick?"""

rprint(qb_tokenization.bling_tokenizer.tokenize(text))

# %%

qb_tokenization = importlib.reload(qb_tokenization)
sanit_q = acf_sanitization.sanitize_question(tossups_by_id[369].question_text)
print(textwrap.fill(sanit_q))
spans = list(qb_tokenization.punkt_sent_tokenizer.span_tokenize(sanit_q))
spans = qb_tokenization.merge_spans_by_imbalanced_quotes(sanit_q, spans)
rprint(spans)


# %%
for c in sorted(all_bf_clues, key=len, reverse=True)[:10]:
    print(c)


# %%
def search_tossup_by_text(text: str):
    session = models.create_session("data/acf-23-24.db")
    return (
        session.query(models.Tossup)
        .filter(models.Tossup.question_text.contains(text))
        .all()
    )


tossup = search_tossup_by_text("Man!")[0]
print("Tossup:")
question_text = acf_sanitization.sanitize_question(tossup.question_text)
print(textwrap.fill(question_text))

print("\nBlingFire Clues:")
bf_spans = qb_tokenization.generate_blingfire_spans(question_text)
for i, span in enumerate(bf_spans, 1):
    print(f"{i}. {question_text[span[0]:span[1]]}")

print("\nOrig Blingfire clues:")
bf_spans = qb_tokenization.bling_tokenizer.span_tokenize(question_text)
for i, span in enumerate(bf_spans, 1):
    print(f"{i}. {question_text[span[0]:span[1]]}")

# %%
import re

# List all tossups with '[.?!]" [A-Z]' pattern in the text.
for tossup_id, tossup in tossups_by_id.items():
    question_text = acf_sanitization.sanitize_question(tossup.question_text)
    if re.search(r"[.?!]\"\s[A-Z]", question_text):
        print()
        print(tossup_id)
        print(textwrap.fill(question_text))
        clues = qb_tokenization.get_clue_spans(
            question_text, tokenization_scheme="best"
        )
        for i, clue in enumerate(clues, 1):
            print(f"{i}. {question_text[clue[0]:clue[1]]}")
        print("-" * 100)

# %%
