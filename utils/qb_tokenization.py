from typing import Iterable, Mapping, Optional, Sequence

from blingfire import text_to_sentences
from nltk.tokenize import PunktSentenceTokenizer

punkt_sent_tokenizer = PunktSentenceTokenizer()

UNICODE_QUOTE_START = "“"
UNICODE_QUOTE_END = "”"

_QUOTES = {'"': '"', UNICODE_QUOTE_START: UNICODE_QUOTE_END}


def find_quote(sent: str, quote: Optional[str] = None):
    # Note: This doesn't handle nested quotes.
    # Following Ostrich Algorithm (https://en.wikipedia.org/wiki/Ostrich_algorithm) :D
    if quote is not None:
        count = len([ch for ch in sent if ch == quote])
        return quote if count % 2 else None

    for q in _QUOTES:
        q = find_quote(sent, q)
        if q:
            return q

    return None


def merge_quote_spans(tokenizations: list, text: str, verbose: bool = False):
    current_quote = None
    merged_tokenizations = []
    for s_new, e_new in tokenizations:
        if current_quote:  # within quotes currently
            new_quote = find_quote(text[s_new:e_new], _QUOTES[current_quote])
            if not new_quote:
                continue
            if verbose:
                print(f"Found {new_quote} for {current_quote}")
            # merge all pending spans till now
            s_top, e_top = merged_tokenizations[-1]
            merged_tokenizations[-1] = s_top, e_new
            current_quote = None

        else:  #
            current_quote = find_quote(text[s_new:e_new])
            merged_tokenizations.append((s_new, e_new))

    if current_quote:
        s, e = merged_tokenizations[-1]
        raise RuntimeError(f"Error while processing question clue: \n{text[s:e]}")

    return merged_tokenizations


def generate_punkt_sent_spans(
    text: str,
    sent_tokenizer=punkt_sent_tokenizer,
    verbose: bool = False,
    return_sents=False,
):
    tokenizations = list(sent_tokenizer.span_tokenize(text))
    merged_tokenizations = [tokenizations[0]]
    for i in range(1, len(tokenizations)):
        curr_start, curr_end = merged_tokenizations[-1]
        new_start, new_end = tokenizations[i]
        if text[new_start].islower():
            if verbose:
                print(
                    f"Undesired split detected, merging the spans {(curr_start, curr_end)} and {(new_start, new_end)}."
                )
                print(f"Current clue: {text[curr_start:curr_end]}")
                print(f"New clue    : {text[new_start:new_end]}")
            merged_tokenizations[-1] = curr_start, new_end
        else:
            merged_tokenizations.append((new_start, new_end))
    merged_tokenizations = merge_quote_spans(
        merged_tokenizations, text, verbose=verbose
    )

    if return_sents:
        sents = [text[s:e] for s, e in merged_tokenizations]
        return merged_tokenizations, sents
    return [tuple(t) for t in merged_tokenizations]


def get_spans_from_sents(text: str, sents: Iterable[str]):
    sents = [s.strip() for s in sents]
    tokenizations = [[0, len(sents[0])]]
    for sent in sents[1:]:
        s = tokenizations[-1][1]
        look_ahead_limit = 20
        while look_ahead_limit > 0 and s < len(text) and text[s] != sent[0]:
            s += 1
            look_ahead_limit -= 1
        if look_ahead_limit == 0:
            raise RuntimeError("Lookahead limit reached.")
        e = s + len(sent)
        tokenizations[-1][1] = s
        tokenizations.append([s, e])
    return tokenizations


def apply_spans(text: str, spans: Iterable[Sequence[int]]):
    return [text[s:e] for (s, e) in spans]


def get_blingfire_sents(text: str, spans: bool = False):
    sents = text_to_sentences(text).split("\n")
    if not spans:
        return sents
    return get_spans_from_sents(text, sents)


def generate_blingfire_spans(text: str):
    return get_blingfire_sents(text, spans=True)


def get_clue_spans(qb_dict: Mapping | str, tokenization_scheme: str = "best"):
    if isinstance(qb_dict, str):
        q = qb_dict
        qanta_spans = None
    else:
        q = qb_dict["text"]
        qanta_spans = q.get("tokenizations", None)

    if tokenization_scheme == "qanta":
        if qanta_spans:
            return qanta_spans
        raise RuntimeError(
            "Qanta Tokenizations not found in question dictionary. Please provide a valid question dictionary, or use a different tokenization scheme."
        )

    bf_spans = generate_blingfire_spans(q)

    try:
        punkt_spans = generate_punkt_sent_spans(q)
    except RuntimeError:
        punkt_spans = None

    if tokenization_scheme == "punkt":
        return punkt_spans
    elif tokenization_scheme == "blingfire":
        return bf_spans
    elif tokenization_scheme == "best":
        best_spans = qanta_spans
        if punkt_spans and (not best_spans or (3 < len(punkt_spans) < len(best_spans))):
            best_spans = punkt_spans
        if not best_spans or len(bf_spans) <= len(best_spans) + 1:
            best_spans = bf_spans
        return best_spans
    else:
        raise ValueError(f"Invalid tokenization_scheme '{tokenization_scheme}'.")


def get_clues(qb_dict: Mapping, tokenization_scheme: str = "best"):
    spans = get_clue_spans(qb_dict, tokenization_scheme)
    return apply_spans(qb_dict["text"], spans)


if __name__ == "__main__":
    text = """This is the first sentence, 
    the hardest clue.
    The sentence after the first clue 
    is the second clue. For 10 points, name this 
    entity for which this is a giveaway clue."""

    qb_dict = {"text": text}
    for i, clue in enumerate(get_clues(qb_dict)):
        print(i + 1, repr(clue))
