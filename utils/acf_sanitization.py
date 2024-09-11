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


def remove_pgs(q):
    # Remove Pronunciation Guides
    return re.sub('\\s\\(["“][^)]+[”"]\\)', "", q)


def remove_mod_instructions(q):
    return re.sub("\\s\\[[(emphasize|pause|read slowly)]+\\]", "", q)


def remove_tags(q):
    return re.sub(r"<\/?(em|b|i|u)>", "", q)


def sanitize_question(q):
    q, _ = remove_instruction(q)
    q = convert_html_symbols(q)
    q = remove_mod_instructions(q)
    q = remove_tags(q)
    q = remove_pgs(q)
    q = remove_power_pos(q)
    return q.strip()


def get_buzz_offset(q):
    _, inst = remove_instruction(q)
    return len(inst.split())


def acf_tokenize(q):
    return sanitize_question(q).split()
