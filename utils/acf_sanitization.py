# %%
import re
from typing import Tuple

import datasets


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


def normalize_quotes(q):
    for c in ["\\“", "\\”", '\\"', "“", "”"]:
        q = q.replace(c, '"')
    q = q.replace("\\'", "'")
    return q


def remove_braces(s: str) -> str:
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
    a = normalize_quotes(a)
    a = squish_whitespace(a)
    return a.strip()


def get_buzz_offset(q):
    _, inst = remove_instruction(q)
    return len(inst.split())


def sanitokenize(q):
    return sanitize_question(q).split()


def normalize_html_answer_line(answer_line: str):
    answer_line = answer_line.removeprefix("</b>")
    answer_line = re.sub(r"<u><b>|<b><u>", "<b>", answer_line)
    answer_line = re.sub(r"</u></b>|</b></u>", "</b>", answer_line)
    answer_line = answer_line.replace("<b>", "{").replace("</b>", "}")
    answer_line = remove_tags(answer_line)
    return answer_line


# Check if answer line has explanation. Explanation is in () at the end of the line
def split_explanation(line: str):
    if (j := line.rfind("]")) != -1:
        if (i := line[j:].find("(")) != -1:
            explanation = line[i + j :]
            if remove_pgs(" " + explanation) != "":
                answer_line = line[: i + j].strip()
                explanation = explanation.strip("( )")
                return answer_line, explanation
    if line.endswith(")") and (i := line.rfind("(")) > 0:
        explanation = line[i:]
        if remove_pgs(" " + explanation) != "":
            answer_line = line[:i].strip()
            explanation = explanation.strip("( )")
            return answer_line, explanation
    return line, ""


def extract_possible_answers(braced_answer: str):
    """
    Extract possible answers from a single answer that has {} brackets.
    Text within {} is mandatory, other is optional.
    Extract answers such that a correct answer is an exact match with at least one of the extracted answers.
    Example:
        "{Joe} Biden" -> ["Joe Biden", "Joe"]
        "{J}oseph {Biden}" -> ["Joseph Biden", "J Biden"]
        "{Joe} {Biden}" -> ["Joe Biden"]
        "{J}ohn {F}rank {K}ennedy" -> ["J F K", "J F Kennedy", "John F K", "John F Kennedy", "J Frank K", "J Frank Kennedy", "John Frank K", "John Frank Kennedy"]
    """
    import re
    from itertools import product

    # Handle empty or None input
    if not braced_answer:
        return []

    # Tokenize into segments: each is either a {braced} or a plain run
    pattern = re.compile(r"(\{[^{}]+\})")
    segments = []
    last = 0
    for m in pattern.finditer(braced_answer):
        if m.start() > last:
            segments.append(braced_answer[last : m.start()])
        segments.append(m.group())
        last = m.end()
    if last < len(braced_answer):
        segments.append(braced_answer[last:])

    # Group segments into braced groups and standalone non-braced groups.
    # A braced group is (braced, non_braced_after)
    groups = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if re.fullmatch(r"\{([^{}]+)\}", seg):
            braced_content = re.fullmatch(r"\{([^{}]+)\}", seg).group(1)
            # If there's a following non-braced piece, attach as suffix
            if i + 1 < len(segments) and not re.fullmatch(
                r"\{([^{}]+)\}", segments[i + 1]
            ):
                groups.append((braced_content, segments[i + 1]))
                i += 2
            else:
                groups.append((braced_content, ""))
                i += 1
        else:
            # Non-braced chunk not adjacent to a following braced segment (or trailing text)
            groups.append(("", seg))
            i += 1

    # Build options for each group.
    # For braced groups with a non-braced suffix, include:
    #  - braced-only, but preserve a single separating space if the original suffix contained whitespace
    #  - braced + full suffix
    # For non-braced initial chunk (idx == 0), make it optional: ["", chunk]
    options = []
    for idx, (braced, non_braced) in enumerate(groups):
        if braced:
            # treat any following non_braced string (including pure whitespace) as a suffix
            if non_braced != "":
                # preserve a single separator in the braced-only choice when the suffix has
                # leading or trailing whitespace so subsequent groups don't get concatenated
                sep = (
                    " " if (non_braced[0].isspace() or non_braced[-1].isspace()) else ""
                )
                options.append([braced + sep, braced + non_braced])
            else:
                options.append([braced])
        else:
            if idx == 0 and non_braced.strip():
                # initial non-braced chunk should be optional
                options.append(["", non_braced])
            else:
                options.append([non_braced])

    # Generate combinations by simple concatenation, then normalize whitespace.
    all_combos = []
    for combo in product(*options):
        candidate = "".join(combo)
        candidate = squish_whitespace(candidate).strip()
        if candidate:
            all_combos.append(candidate)

    # Deduplicate and sort by length (descending) for deterministic order
    all_combos = sorted(set(all_combos), key=lambda x: (-len(x), x))
    return all_combos


def _extract_braced_answer_chunks(raw_answer_text: str):
    answer_line, explanation = split_explanation(raw_answer_text)

    answer_line = normalize_quotes(answer_line)
    answer_line = answer_line.replace("&nbsp;", " ")
    answer_line = remove_tags(answer_line)
    answer_line = remove_pgs(answer_line)
    answer_line = answer_line.replace('"', "")

    answers = []
    # Find rejection phrases like [prompt ...], [do not accept ...], [before ...]
    # and remove them and everything after them
    rejection_span_start_indices = list(
        re.finditer(r"(\[| |;)(reject|prompt|do not accept|before) ", answer_line)
    )
    if rejection_span_start_indices:
        idx = rejection_span_start_indices[0].span()[0]
        answer_line = answer_line[:idx]

    if "[" in answer_line:
        gold, alternates_str = answer_line.split("[", 1)
        alternates_str = alternates_str.removesuffix("]")
    else:
        gold, alternates_str = answer_line, ""

    candidates = []

    alternates = alternates_str.removeprefix("or ").removeprefix("accept ")
    # Split alternates on " or " or " or, " and iterate over unique results
    for words in set(re.split(r"(?: or,? |; or )", alternates)):
        candidates.extend(words.split(" accept "))
    for words in set(re.split(r"(?: or,? |; or )", gold)):
        candidates.extend(words.split(" accept "))

    answers.extend(candidates)

    def cleanup_braced_answer(ans: str) -> str:
        ans = ans.removeprefix("}").removesuffix("{").strip()
        return ans.replace("{ }", "").replace("{}", "").strip()

    answers = list({cleanup_braced_answer(a) for a in answers} - {""})
    for a in answers:
        if " AND " in a:
            print(f"Found AND in answer: {raw_answer_text} -> {a}")
            # replace AND with {and}, and also insert "B and A" for "A and B"
            a = a.replace(" AND ", " {and} ")
            answers.append(a)
            parts = a.split(" {and} ")
            if len(parts) == 2:
                a_rev = f"{parts[1]} {{and}} {parts[0]}"
                answers.append(a_rev)
            else:
                print(f"Could not process AND in answer: {raw_answer_text} -> {a}")
        elif " OR " in a:
            print(f"Found OR in answer: {raw_answer_text} -> {a}")
            parts = a.split(" OR ")
            # insert each part as a separate answer
            answers.extend(parts)
    return answers, explanation


def extract_braced_answers(braced_ans_text: str):
    answers = {remove_braces(braced_ans_text)}
    braced = re.findall(r"\{.+?\}", braced_ans_text)  # find all {braced} answers
    if len(braced) >= 1:
        answers.add(" ".join(map(remove_braces, braced)))
        answers.update(map(remove_braces, braced))
    return list(answers - {""})


# TODO(maharshi95): Handle "or word forms such as" and "or synonyms such as", maybe more generally "
# "or <noun phrase> such as <comma separated examples>"
def get_clean_answers(raw_ans_text: str, primary: bool = True):
    raw_ans_text = squish_whitespace(raw_ans_text).strip()
    answer_line, explanation = split_explanation(raw_ans_text)

    answer_line = normalize_quotes(answer_line)
    answer_line = answer_line.replace("&nbsp;", " ")
    answer_line = remove_tags(answer_line)
    answer_line = remove_pgs(answer_line)
    answer_line = answer_line.replace('"', "")

    answers = []
    # Find rejection phrases like [prompt ...], [do not accept ...], [before ...]
    # and remove them and everything after them
    rejection_span_start_indices = list(
        re.finditer(r"(\[| |;)(reject|prompt|do not accept|before) ", answer_line)
    )
    if rejection_span_start_indices:
        idx = rejection_span_start_indices[0].span()[0]
        answer_line = answer_line[:idx]

    if "[" in answer_line:
        gold, alternates_str = answer_line.split("[", 1)
        alternates_str = alternates_str.removesuffix("]")
    else:
        gold, alternates_str = answer_line, ""

    candidates = []

    alternates = alternates_str.removeprefix("or ").removeprefix("accept ")
    # Split alternates on " or " or " or, " and iterate over unique results
    for words in set(re.split(r"(?: or,? |; or )", alternates)):
        candidates.extend(words.split(" accept "))
    for words in set(re.split(r"(?: or,? |; or )", gold)):
        candidates.extend(words.split(" accept "))

    answers.extend(candidates)
    for split in candidates:
        braced = re.findall(r"\{.+?\}", split)  # find all {braced} answers
        if len(braced) >= 1:
            answers.append(" ".join(map(remove_braces, braced)))
            answers.extend(map(remove_braces, braced))

    answers = {*map(remove_braces, answers)} - {""}
    if primary:
        return gold.strip(), list(answers), explanation
    else:
        return list(answers), explanation


def get_short_clean_answers(raw_answer_string: str, max_tokens: int = 10):
    # Replace bold+underline with just bold (usually the bold text is almost always underlined)
    normalized_answer_line = normalize_html_answer_line(raw_answer_string)
    normalized_answer_line = sanitize_answer(normalized_answer_line)
    normalized_answer_line = (
        normalized_answer_line.replace("{ ", "{").replace(" }", "}").replace("{}", "")
    )
    answer_primary, clean_answers, explanation = get_clean_answers(
        normalized_answer_line, primary=True
    )
    clean_answers = set()
    braced_chunks, explanation = _extract_braced_answer_chunks(normalized_answer_line)
    for chunk in braced_chunks:
        clean_answers.update(extract_possible_answers(chunk))
    clean_answers_filtered = [a for a in clean_answers if len(a.split()) <= max_tokens]
    if clean_answers_filtered:
        clean_answers = clean_answers_filtered
    # complex_braced = [b for b in braced_chunks if b.count("{") > 1]
    # if len(complex_braced) >= 1:
    #     print(raw_answer_string)
    #     print(f"Complex braced answers: {complex_braced}")
    #     print(clean_answers)
    #     print()

    return {
        "primary": answer_primary,
        "normalized": normalized_answer_line,
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

    test_cases_brace_answer_extraction = [
        # Basic cases from the docstring
        ("wild{fire}", ["wildfire", "fire"]),
        ("{Joe} Biden", ["Joe Biden", "Joe"]),
        ("{J}oseph {Biden}", ["Joseph Biden", "J Biden"]),
        ("{Joe} {Biden}", ["Joe Biden"]),
        (
            "{J}ohn {F}rank {K}ennedy",
            [
                "John Frank Kennedy",
                "John Frank K",
                "John F Kennedy",
                "John F K",
                "J Frank Kennedy",
                "J Frank K",
                "J F Kennedy",
                "J F K",
            ],
        ),
        # Non-braced text at beginning
        (
            "President {Joe} Biden",
            ["President Joe Biden", "President Joe", "Joe Biden", "Joe"],
        ),
        (
            "The {United} {States} of America",
            [
                "The United States of America",
                "The United States",
                "United States of America",
                "United States",
            ],
        ),
        # Multiple consecutive braced segments
        ("{United} {States} {of} {America}", ["United States of America"]),
        ("{George} {Herbert} {Walker} {Bush}", ["George Herbert Walker Bush"]),
        # Whitespace handling
        ("{Joe} ", ["Joe"]),
        (" {Joe}", ["Joe"]),
        ("{Joe}  {Biden}", ["Joe Biden"]),
        # Mixed cases
        (
            "The {J}ames {K}. {P}olk",
            [
                "The James K. Polk",
                "The James K. P",
                "The James K Polk",
                "The James K P",
                "The J K. Polk",
                "The J K. P",
                "The J K Polk",
                "The J K P",
                "James K. Polk",
                "James K. P",
                "James K Polk",
                "James K P",
                "J K. Polk",
                "J K. P",
                "J K Polk",
                "J K P",
            ],
        ),
        ("{Franklin} D. {Roosevelt}", ["Franklin D. Roosevelt", "Franklin Roosevelt"]),
        # Edge cases
        ("No braces here", ["No braces here"]),
        ("{All} {braced} {text}", ["All braced text"]),
        ("{Pope} {John} {Paul} II", ["Pope John Paul II", "Pope John Paul"]),
    ]
    for input_text, expected in test_cases_brace_answer_extraction:
        result = extract_possible_answers(input_text)
        if set(result) != set(expected):
            print(f"Failed for '{input_text}'")
            if missed := set(expected) - set(result):
                print(f"Missed: {missed}")
            if undesired := set(result) - set(expected):
                print(f"Undesired: {undesired}")
    # %%

    dataset_name = "qanta-challenge/qanta25-final"
    bonus_ds = datasets.load_dataset(dataset_name, "bonus", split="eval")
    tossup_ds = datasets.load_dataset(dataset_name, "tossup", split="eval")
    raw_answers = []
    for e in tossup_ds:
        raw_answers.append(e["answer_line"])
    for e in bonus_ds:
        for p in e["parts"]:
            raw_answers.append(p["answer_line"])
    # %%
    all_clean_answers = []
    print("# raw answer strings:", len(raw_answers))
    for a in raw_answers:
        clean_answers = get_short_clean_answers(a)["clean"]
        if "Guinea" in a:
            print(a, clean_answers)

        # short_answers = [c for c in clean_answers if len(c.split()) >= 10]
        # if short_answers:
        #     print(a, short_answers, end="\n\n", sep="\n")
        #     print(clean_answers)
        # all_clean_answers.extend(clean_answers)
    print("# all clean answer strings:", len(all_clean_answers))

    # %%
