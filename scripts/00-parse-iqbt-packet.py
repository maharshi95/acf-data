import argparse
import json
import os
import re

import fitz

StrDict = dict[str, str]


def extract_with_html_tags(pdf_path: str) -> str:
    """
    Extracts all text from the PDF at pdf_path, wrapping
    bold runs in <b>, italics in <i>, underlines in <u>.
    Returns a big HTML-annotated string.
    """
    doc = fitz.open(pdf_path)
    out_lines = []

    # bitmask constants (from MuPDF)
    ITALIC_FLAG = 1 << 1  # 2
    BOLD_FLAG = 1 << 6  # 64
    UNDERLINE_FLAG = 1 << 0  # 1 (not always reliable)

    for page in doc:
        data = page.get_text("dict")  # get the blocks→lines→spans structure
        for block in data["blocks"]:
            if block["type"] != 0:  # skip non-text blocks
                continue
            for line in block["lines"]:
                line_html = ""
                for span in line["spans"]:
                    txt = span["text"]
                    font = span["font"]
                    flags = span["flags"]

                    # detect styles via flags *or* font name
                    is_bold = bool(flags & BOLD_FLAG) or ("Bold" in font)
                    is_italic = (
                        bool(flags & ITALIC_FLAG)
                        or ("Italic" in font)
                        or ("Oblique" in font)
                    )
                    is_underline = bool(flags & UNDERLINE_FLAG) or ("Underline" in font)

                    # wrap in tags, nesting: <u><b><i>text</i></b></u>
                    wrapped = txt
                    if is_italic:
                        wrapped = f"<i>{wrapped}</i>"
                    if is_bold:
                        wrapped = f"<b>{wrapped}</b>"
                    if is_underline:
                        wrapped = f"<u>{wrapped}</u>"

                    line_html += wrapped

                out_lines.append(line_html)

    return "\n".join(out_lines)


def parse_tossup_block(block: str) -> list[StrDict]:
    # split on question numbers at start of line: "1. ", "2. ", …
    raw_qs = re.split(r"\n(?=\d+\.\s)", block)

    questions = []
    for raw in raw_qs:
        raw = raw.strip()
        if not raw or not raw[0].isdigit():
            continue

        # capture: "<num>. QUESTION… ANSWER: … Category: …"
        m = re.match(
            r"(?P<number>\d+)\.\s+"  # question number
            r"(?P<question>.*?)"  # question text (non-greedy)
            r"ANSWER:\s*(?P<answer>.*?)\s*"  # answer text (non-greedy)
            r"Category:\s*(?P<category>.+)",  # category (rest of line)
            raw,
            flags=re.S | re.I,
        )
        if not m:
            # skip anything that doesn't match
            continue

        q = {
            "question_number": int(m.group("number")),
            "question_raw": m.group("question").strip(),
            "answer_raw": m.group("answer").strip(),
            "category": m.group("category").split("\n")[0].strip(),
        }
        questions.append(q)

    return questions


def parse_bonus_block(block: str) -> list[StrDict]:
    """
    Parse a block of bonus questions, extracting the lead-in, parts, and answers.

    Args:
        block (str): Text block containing bonus questions

    Returns:
        list[dict[str, str]]: List of bonus questions with their parts and answers
    """
    # split on question numbers at start of line: "1. ", "2. ", …
    raw_qs = re.split(r"\n(?=\d+\.\s)", block)

    questions = []
    for raw in raw_qs:
        raw = raw.strip()
        if not raw or not raw[0].isdigit():
            continue

        # Split the question into lead-in and parts
        parts = re.split(r"\n(?=\[10\])", raw)
        if not parts:
            continue

        # Parse the lead-in (first part)
        lead_in = parts[0]
        m = re.match(
            r"(?P<number>\d+)\.\s+"  # question number
            r"(?P<lead_in>.*?)"  # lead-in text (non-greedy)
            r"For\s+10\s+points\s+each:",  # points indicator
            lead_in,
            flags=re.S | re.I,
        )

        if not m:
            continue

        # Parse the bonus parts
        bonus_parts = []
        for part in parts[1:]:
            part_match = re.match(
                r"\[10\]\s+"  # points marker
                r"(?P<part_text>.*?)"  # part text (non-greedy)
                r"ANSWER:\s*(?P<part_answer>.*?)(?=\n\[10\]|\nCategory:|\Z)",  # part answer
                part,
                flags=re.S | re.I,
            )
            if part_match:
                bonus_parts.append(
                    {
                        "part_text": part_match.group("part_text").strip(),
                        "part_answer": part_match.group("part_answer").strip(),
                    }
                )

        # Extract category from the last part
        category_match = re.search(r"Category:\s*(.+)$", raw, flags=re.S | re.I)
        category = category_match.group(1).strip() if category_match else ""

        q = {
            "question_number": int(m.group("number")),
            "lead_in": m.group("lead_in").strip(),
            "category": category,
            "parts": bonus_parts,
        }
        questions.append(q)

    return questions


def parse_packet_content(
    html: str, ignores: list[str] = []
) -> dict[str, list[StrDict]]:
    """
    Given the full HTML-tagged text of the packet, returns a tuple of (tossups, bonuses):
    tossups: [{
        "question_number": int,
        "question_raw": str,
        "answer_raw": str,
        "category": str
    }, ...]
    bonuses: [{
        "question_number": int,
        "lead_in": str,
        "category": str,
        "parts": [{
            "part_text": str,
            "part_answer": str
        }, ...]
    }, ...]
    """
    # isolate the tossup section
    # (assumes there's a header like "<b>Round 1 – Tossups</b>")

    for ignore_pattern in ignores:
        html = html.replace(ignore_pattern, "")

    html = html.strip()

    # Split on either "Round {number} – Tossups" or "TOSSUPS" pattern
    tossup_parts = re.split(
        r"<b>(?:Round\s+\d+\s+–\s+Tossups|TOSSUPS)\s*</b>", html, flags=re.I
    )

    # If we have bonus sections, split further
    if len(tossup_parts) > 1:
        # Get the part after the tossup header
        questions_section = tossup_parts[1]
        # Check if there's a bonus section - match either pattern
        bonus_parts = re.split(
            r"<b>(?:Round\s+\d+\s+–\s+Bonuses|BONUSES)\s*</b>",
            questions_section,
            flags=re.I,
        )
        parts = bonus_parts
    else:
        parts = tossup_parts
    if len(parts) < 2:
        raise ValueError(f"Only found {len(parts)} sections in packet")
    tossup_block, bonus_block = parts[:2]

    tossups = parse_tossup_block(tossup_block)
    bonuses = parse_bonus_block(bonus_block)

    return {
        "tossups": tossups,
        "bonuses": bonuses,
    }


# %%

terms = """© International Quiz Bowl Tournaments, LLC 
Terms for usage & distribution of these questions are found in the IQBT host licensing agreement. Questions 
may not be used for profit without permission from IQBT. Questions may not be used for artificial intelligence 
or machine learning training."""


packets = {
    "2024 IQBT National Packet 01": "data/packets/2024 IQBT National – Packet 01.pdf",
    "2024 IQBT National Packet 02": "data/packets/2024 IQBT National – Packet 02.pdf",
    "2024 IQBT National Packet 03": "data/packets/2024 IQBT National – Packet 03.pdf",
}


def main(filepath: str, packet_name: str):
    text = extract_with_html_tags(filepath)
    text = text.replace(terms, "").strip()
    packet_content = parse_packet_content(text, [terms])
    for t in packet_content["tossups"]:
        t["packet_name"] = packet_name
    for b in packet_content["bonuses"]:
        b["packet_name"] = packet_name
    return packet_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        required=True,
        help="Path to the packet file in pdf format",
    )
    parser.add_argument(
        "--packet", "-p", type=str, default=None, help="Name of the packet"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/",
        help="Path to the output directory where the dataset will be saved",
    )
    args = parser.parse_args()
    packet_name = args.packet or args.filepath.split("/")[-1].rsplit(".", 1)[0]
    packet_content = main(args.filepath, packet_name)
    for qtype in ["tossups", "bonuses"]:
        out_filepath = os.path.join(args.output_dir, f"{packet_name}/{qtype}.jsonl")
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w") as f:
            for q in packet_content[qtype]:
                f.write(json.dumps(q) + "\n")


# %%
