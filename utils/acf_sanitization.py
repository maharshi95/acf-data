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


def tokenize(q):
    return sanitize_question(q).split()


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
