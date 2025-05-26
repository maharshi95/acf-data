# %%
import argparse
import json
import os
import re
from typing import TypedDict

import fitz
from datasets import Dataset
from tqdm import tqdm

from core.structs import BonusPart, QBBonusQuestion, QBTossupQuestion, QuestionMetadata
from utils import acf_sanitization, qb_tokenization
from utils.acf_sanitization import squish_whitespace
from utils.tossups import prepare_token_indices

PREFIX = "2024-iqbt-nats"


class BonusPartDict(TypedDict):
    part_number: int
    part_text: str
    part_answer: str


class BonusQuestionDict(TypedDict):
    question_number: int
    lead_in: str
    category: str
    parts: list[BonusPartDict]


class TossupQuestionDict(TypedDict):
    question_number: int
    question_raw: str
    answer_raw: str
    category: str


def transform_bonus_question(
    bonus: BonusQuestionDict,
    qid_prefix: str,
    question_set: str,
    packet_number: int,
    packet_name: str,
) -> QBBonusQuestion:
    parts = []
    for part in bonus["parts"]:
        raw_answer_str = squish_whitespace(part["part_answer"])
        answers = acf_sanitization.get_short_clean_answers(raw_answer_str)
        part_text = acf_sanitization.sanitize_question(part["part_text"])

        bonus_part = BonusPart(
            number=part["part_number"],
            question=part_text,
            answer=raw_answer_str,
            answer_primary=answers["primary"],
            clean_answers=answers["clean"],
            explanation=answers["explanation"],
            value=10,
            difficulty_modifier="",
        )
        parts.append(bonus_part)

    leadin = acf_sanitization.sanitize_question(bonus["lead_in"])
    return QBBonusQuestion(
        qid=f"{qid_prefix}-{packet_number:02d}-{bonus['question_number']}",
        leadin=leadin,
        parts=parts,
        metadata=QuestionMetadata(
            category=bonus["category"],
            subcategory=[],
            category_main="",
            category_full=bonus["category"],
            difficulty="",
            question_set=question_set,
            packet=packet_name,
        ),
    )


def transform_tossup_question(
    tossup: TossupQuestionDict,
    qid_prefix: str,
    question_set: str,
    packet_number: int,
    packet_name: str,
) -> QBTossupQuestion:
    answer_raw = squish_whitespace(tossup["answer_raw"])
    question_raw = tossup["question_raw"]
    answers = acf_sanitization.get_short_clean_answers(answer_raw)
    question_sanitized = acf_sanitization.sanitize_question(question_raw)
    clue_spans = qb_tokenization.get_clue_spans(
        question_sanitized, tokenization_scheme="blingfire"
    )
    return QBTossupQuestion(
        qid=f"{qid_prefix}-{packet_number:02d}-{tossup['question_number']}",
        answer=answer_raw,
        clean_answers=answers["clean"],
        explanation=answers["explanation"],
        answer_primary=answers["primary"],
        clue_spans=clue_spans,
        question=question_sanitized,
        metadata=QuestionMetadata(
            category=tossup["category"],
            subcategory=[],
            category_main="",
            category_full=tossup["category"],
            difficulty="",
            question_set=question_set,
            packet=packet_name,
            human_buzz_positions=[],
        ),
    )


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


def parse_tossup_block(block: str) -> list[TossupQuestionDict]:
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

        q = TossupQuestionDict(
            question_number=int(m.group("number")),
            question_raw=m.group("question"),
            answer_raw=m.group("answer"),
            category=m.group("category").split("\n")[0].strip(),
        )
        questions.append(q)

    return questions


def parse_bonus_block(block: str) -> list[BonusQuestionDict]:
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
        raw_question_str = parts[0]
        m = re.match(
            r"(?P<number>\d+)\.\s+"  # question number
            r"(?P<lead_in>.*?)"  # lead-in text (non-greedy)
            r"For\s+10\s+points\s+each:",  # points indicator
            raw_question_str,
            flags=re.S | re.I,
        )

        if not m:
            continue

        # Parse the bonus parts
        bonus_parts = []
        part_number = 0
        for part in parts[1:]:
            part_match = re.match(
                r"\[10\]\s+"  # points marker
                r"(?P<part_text>.*?)"  # part text (non-greedy)
                r"ANSWER:\s*(?P<part_answer>.*?)(?=\n\[10\]|\nCategory:|\Z)",  # part answer
                part,
                flags=re.S | re.I,
            )
            if part_match:
                part_number += 1
                bonus_part = BonusPartDict(
                    part_number=part_number,
                    part_text=part_match.group("part_text"),
                    part_answer=part_match.group("part_answer"),
                )
                bonus_parts.append(bonus_part)

        # Extract category from the last part
        category_match = re.search(r"Category:\s*(.+)$", raw, flags=re.S | re.I)
        category = category_match.group(1).strip() if category_match else ""

        # Extract lead-in
        q = BonusQuestionDict(
            question_number=int(m.group("number")),
            lead_in=m.group("lead_in"),
            category=category,
            parts=bonus_parts,
        )
        questions.append(q)

    return questions


def parse_packet_content(
    html: str, ignores: list[str] = []
) -> dict[str, list[TossupQuestionDict] | list[BonusQuestionDict]]:
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


def main(
    filepath: str,
    question_set: str,
    packet_number: int,
    packet_name: str,
    qid_prefix: str = PREFIX,
):
    text = extract_with_html_tags(filepath)
    text = text.replace(terms, "").strip()
    packet_content = parse_packet_content(text, [terms])
    tossups = []
    bonuses = []
    for t in packet_content["tossups"]:
        t = transform_tossup_question(
            t, qid_prefix, question_set, packet_number, packet_name
        ).to_dict()
        clue_token_indices, run_token_indices = prepare_token_indices(
            t["question"], t["clue_spans"]
        )
        t["clue_token_indices"] = clue_token_indices
        t["run_indices"] = run_token_indices
        tossups.append(t)
    for b in packet_content["bonuses"]:
        b = transform_bonus_question(
            b, qid_prefix, question_set, packet_number, packet_name
        )
        bonuses.append(b.to_dict())
    return {"tossups": tossups, "bonuses": bonuses}


def write_dataset(
    dataset_entries: dict[str, list[TossupQuestionDict] | list[BonusQuestionDict]],
    output_dir: str,
    hf_repo_id: str | None = None,
):
    for qtype in ["tossups", "bonuses"]:
        out_filepath = os.path.join(output_dir, f"{qtype}.jsonl")
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w") as f:
            for q in dataset_entries[qtype]:
                f.write(json.dumps(q) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the packet file in pdf format, or a directory containing multiple packet files",
    )
    parser.add_argument(
        "--packet",
        "-p",
        type=str,
        default=None,
        help="Name of the packet. Should not be provided if --input is a directory. Optional otherwise.",
    )
    parser.add_argument(
        "--question-set",
        "-q",
        type=str,
        default="2024-iqbt-nationals",
        help="Question set name",
    )
    parser.add_argument(
        "--packet-id",
        "-i",
        type=str,
        default=None,
        help="Packet ID. Should not be set if --input is a directory, required otherwise.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/datasets/",
        help="Path to the output directory where the dataset will be saved",
    )

    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID. If provided, the dataset will be pushed to the Hugging Face Hub.",
    )
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.question_set)

    # Validate arguments
    if not os.path.exists(args.input):
        raise ValueError(f"Input file or directory {args.input} does not exist")

    if os.path.isfile(args.input):
        if not args.input.endswith(".pdf"):
            raise ValueError(f"Input file {args.input} is not a PDF")

        if args.packet_id is None:
            raise ValueError(
                "Packet ID must be provided if input is a single packet file"
            )
        if args.hf_repo_id is not None:
            raise ValueError(
                "Currently, we do not support pushing to the Hugging Face Hub when "
                "input is a single packet file. Please put all the packets in a "
                "directory and run the script on the directory to create a single "
                "dataset and push that to the Hub."
            )
        packet_name = args.packet or args.filepath.split("/")[-1].rsplit(".", 1)[0]
        packet_content = main(
            args.filepath, args.packet_id, args.question_set, packet_name
        )
        write_dataset(packet_content, output_dir)

    else:
        if args.packet_id is not None:
            raise ValueError("Packet ID should not be provided if input is a directory")
        if args.packet is not None:
            raise ValueError(
                "Packet name should not be provided if input is a directory"
            )

        filenames = os.listdir(args.input)
        filenames.sort()

        dataset_entries = {"bonuses": [], "tossups": []}

        for packet_idx, filename in tqdm(
            enumerate(filenames, start=1),
            total=len(filenames),
            desc="Processing packet files",
        ):
            filepath = os.path.join(args.input, filename)
            packet_name = filename.split(".")[0]

            if not filepath.endswith(".pdf"):
                print(f"Skipping {filepath} because it is not a PDF")
                continue
            packet_id = filepath.split(".")[0]
            packet_content = main(
                filepath,
                question_set=args.question_set,
                packet_number=packet_idx,
                packet_name=packet_name,
            )
            dataset_entries["bonuses"].extend(packet_content["bonuses"])
            dataset_entries["tossups"].extend(packet_content["tossups"])

        write_dataset(dataset_entries, output_dir)

        if args.hf_repo_id is not None:
            tossup_dataset = Dataset.from_list(dataset_entries["tossups"])
            tossup_dataset.push_to_hub(
                f"{args.hf_repo_id}", config_name="tossups", split="eval"
            )

            bonus_dataset = Dataset.from_list(dataset_entries["bonuses"])
            bonus_dataset.push_to_hub(
                f"{args.hf_repo_id}", config_name="bonuses", split="eval"
            )


# %%
