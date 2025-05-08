# %%
import re
from typing import Tuple


def remove_instruction(q: str) -> Tuple[str, str]:
    # Check if q starts with <em>..</em> if so, check if it contains a sentence:
    # there is a period before or after the </em>
    inst = ""
    if q.startswith("<em>"):
        i = q.find("</em>")
        start = i + len("</em>")
        if q[start] == ".":
            start += 1

        # Extract the text inside the <em> tag
        # remove all <"/u/i> tags"""
        inst = re.sub(r"<\/?(em|b|i|u)>", "", q[:start]).strip()

        # Check if the instruction ends with a period
        if inst.startswith("Note to") or inst.endswith(".") or q[i + 5] == ".":
            q = q[start:].strip()
    return q, inst


def convert_html_symbols(q):
    html_entity_map = {
        "&nbsp;": " ",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&amp;": "&",
    }
    for entity, replacement in html_entity_map.items():
        q = q.replace(entity, replacement)
    return q


def remove_power_pos(q):
    return q.replace("(*)", "")


def squish_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_pgs(q):
    # Remove Pronunciation Guides
    # This function catches the following patterns:
    # 1. Any characters that are not a closing parenthesis
    # 2. Two double quotes (either straight or curly)
    # 3. A closing parenthesis
    # 4. Two double quotes (either straight or curly)

    # Examples of patterns caught:
    # '(" pro-NUN-see-AY-shun ")
    # '(" PRO-nun-see-AY-shun ")
    # q = re.sub('\\s\\([""][^)]+[""]\\)', "", q)

    # Example: (("Cow-wet")), (( Cow-wet )), ((“Cow-wet”)) (("even no hyphens"))
    q = re.sub(r'\(\([“" ][^)]+[”" ]\)\)', "", q)

    # Example: [["some text"]], [[some text]]
    q = re.sub(r'\[\[[“" ][^)]+[”" ]\]\]', "", q)

    # Example: ((some-text)), ((YES-beh-ray)), but not ((someText)) or ((sometext))
    q = re.sub(r"\(\([^\s)]*-[^\s)]*\)\)", "", q)

    # Example: [[some-text]], [[some-TEST]], but not [[someText]] or [[some text]]
    q = re.sub(r"\[\[[^\s)]*-[^\s)]*\]\]", "", q)

    # Example: (“Cow-wet”), ("Cow-wet"), ("even no hyphens"), (“even no hyphens”)
    q = re.sub(r'\s\(["“][^)]+[”"]\)', "", q)

    # Example: [“Cow-wet”], [“Cow-wet”], [“even no hyphens”], [“even no hyphens”]
    q = re.sub(r'\s\[["“][^)]+[”"]\]', "", q)

    # Example: (some-text), (YES-beh-ray), but not (someText) or (sometext)
    q = re.sub(r"\s\(\"?[^\s)]*-[^\s)]*\)", "", q)

    # Example: [some-text], [some-TEST], but not [someText] or [some text]
    q = re.sub(r"\[[^\s)]*-[^\s)]*\]", "", q)

    return q


def remove_mod_instructions(q):
    """
    Remove moderator instructions from the question text.

    This function removes patterns like:
    - [emphasize]
    - [pause]
    - [read slowly]
    - (emphasize)
    - (pause)
    - (read slowly)
    - [read slowly to end of sentence]

    These instructions are typically enclosed in square brackets or parentheses and
    may appear anywhere in the question text, potentially with text before them.

    Args:
        q (str): The input question text.

    Returns:
        str: The question text with moderator instructions removed.
    """
    # return re.sub("\\s\\[[(emphasize|pause|read slowly)]+\\]", "", q)

    # Remove standard moderator instructions
    q = re.sub(r"(\S*\s*)?[\[(](emphasize|pause|read slowly)[\])]", r"\1", q)

    # Remove [read slowly to end of sentence]
    q = re.sub(r"\[read slowly to end of sentence\]", "", q)

    return q


def remove_tags(q):
    return re.sub(r"<\/?(em|b|i|u)>", "", q)


def sanitize_question(q):
    q = squish_whitespace(q)
    q, _ = remove_instruction(q)
    q = convert_html_symbols(q)
    q = remove_mod_instructions(q)
    q = remove_tags(q)
    q = remove_pgs(q)
    q = remove_power_pos(q)
    return q.strip()


def sanitize_answer(a):
    a = squish_whitespace(a)
    a = convert_html_symbols(a)
    a = remove_tags(a)
    a = remove_pgs(a)
    a = remove_power_pos(a)
    return a.strip()


def get_buzz_offset(q):
    _, inst = remove_instruction(q)
    return len(inst.split())


def sanitokenize(q):
    return sanitize_question(q).split()


# Check if answer line has explanation. Explanation is in () at the end of the line
def split_explanation(line: str):
    if line.endswith(")") and (i := line.rfind("(")) > 0:
        explanation = line[i:]
        if remove_pgs(" " + explanation) != "":
            answer_line = line[:i].strip()
            explanation = explanation.strip("()")
            return answer_line, explanation
    return line, ""


def get_clean_answers(raw_ans_text: str, primary: bool = True):
    raw_ans_text = squish_whitespace(raw_ans_text)
    answer_line, explanation = split_explanation(raw_ans_text)

    for c in ["\\“", "\\”", '\\"']:
        answer_line = answer_line.replace(c, '"')
    for c in ["“", "”"]:
        answer_line = answer_line.replace(c, '"')
    answer_line = answer_line.replace("\\'", "'")
    answer_line = answer_line.replace("&nbsp;", " ")
    answer_line = remove_tags(answer_line)
    answer_line = remove_pgs(answer_line)
    answer_line = answer_line.replace('"', "")

    def normalize_braces(s: str) -> str:
        s = (
            s.replace("{", "")
            .replace("}", "")
            .removeprefix("or ")
            .removesuffix("]")
            .removesuffix(")")
            .removesuffix(";")
            .strip()
        )
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s

    answers = []
    bad_index_starts = list(
        re.finditer(r"(\[| |;)(prompt|do not accept|before) ", answer_line)
    )
    if bad_index_starts:
        idx = bad_index_starts[0].span()[0]
        answer_line = answer_line[:idx]

    if "[" in answer_line:
        gold, alternates = answer_line.split("[", 1)
    else:
        gold, alternates = answer_line, ""

    candidates = []

    alternates = alternates.removeprefix("or ").removeprefix("accept ")
    for words in set(re.split(r" or,? ", alternates)):
        candidates.extend(words.split(" accept "))
    for words in set(re.split(r" or,? ", gold)):
        candidates.extend(words.split(" accept "))

    answers.extend(candidates)
    for split in candidates:
        braced = re.findall(r"\{.+?\}", split)  # find all {braced} answers
        if len(braced) >= 1:
            answers.append(" ".join(map(normalize_braces, braced)))
            answers.extend(map(normalize_braces, braced))

    answers = {*map(normalize_braces, answers)} - {""}
    if primary:
        return gold.strip(), list(answers), explanation
    else:
        return list(answers), explanation


def get_short_clean_answers(raw_answer_string: str, max_tokens: int = 10):
    answer = re.sub(r"<u><b>|<b><u>", "<b>", raw_answer_string)
    answer = re.sub(r"</u></b>|</b></u>", "</b>", answer)
    answer = answer.replace("<b>", "{").replace("</b>", "}")
    answer = sanitize_answer(answer)
    answer_primary, clean_answers, explanation = get_clean_answers(answer, primary=True)
    print(answer_primary, clean_answers, explanation)
    clean_answers = [a for a in clean_answers if len(a.split()) <= max_tokens]
    return {
        "primary": answer_primary,
        "clean": clean_answers,
        "explanation": explanation,
    }


if __name__ == "__main__":
    texts = [
        "hello ((HEL-low))",
        'hello (("HEL-low"))',
        "hello (( HEL-low ))",
        "hello ((“HEL-low”))",
        "hello [[HEL-low]]",
        'hello [["HEL-low"]]',
        "hello [[ HEL-low ]]",
        "hello [[“HEL-low”]]",
        "hello [[HELlow]]",
        'cow (" cow-DEE-yo")',
        "hello (( hello-HEE-loh ))",
        "Covet (“Cow-wet”)",
        'Covet ("Cow-wet")',
        "((“Cow-wet”))",
        '("cow-DEE-yo")',
        '(("Cow-wet"))',
        "(YOSS-beh-ray)",
        "[YOSS-beh-ray]",
        "(YOSSbehRray)",
        "[YOSSbehRray]",
        "Jaconbang ((YOO-kohn-baong))",
        "Jaconbang [[YOO-kohn-baong]]",
    ]
    for text in texts:
        print(remove_pgs(text))
