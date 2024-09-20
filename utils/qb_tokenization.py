# %%
from typing import Iterable, Mapping, Optional, Sequence

import blingfire
from nltk.tokenize import PunktSentenceTokenizer

punkt_sent_tokenizer = PunktSentenceTokenizer()

UNICODE_QUOTE_START = "“"
UNICODE_QUOTE_END = "”"

_QUOTES = {'"': '"', UNICODE_QUOTE_START: UNICODE_QUOTE_END}


def apply_spans(text: str, spans: Iterable[Sequence[int]]):
    return [text[s:e] for (s, e) in spans]


# Low level utils
def is_quote_open(sent: str, quote: str):
    """Checks if the input sentence is balanced with respect to the quote."""
    # Symmetric quotes
    if quote == _QUOTES[quote]:
        return sent.count(quote) % 2 == 1
    else:
        return sent.count(quote) - sent.count(_QUOTES[quote]) > 0


def find_any_unbalanced_start_quote(sent: str):
    """Find any unbalanced start quote in the sentence."""
    quotes = [q for q in _QUOTES if is_quote_open(sent, q)]
    if quotes:
        assert len(quotes) == 1, f"Found multiple quotes in a '{sent}': {quotes}"
        return quotes[0]
    return None


class BlingSentTokenizer:
    """Wrapper Class that follows the PunktSentenceTokenizer API."""

    def get_spans_from_sents(self, text: str, sents: Iterable[str]):
        spans = []
        curr_start = 0
        for sent_i, sent in enumerate(sents):
            sent = sent.strip()
            if not sent:
                continue
            i = text.find(sent, curr_start)
            if i == -1:
                raise RuntimeError(
                    f"Sentence not found. \nSentence: {sent}\nText: {text[curr_start:]}"
                )
            if sent.startswith("/"):
                # Merge the spans i and i+1 if span[i+1] starts with "/"
                spans[-1][1] = i + len(sent)
            elif sent_i > 0 and sent[0].isupper() and sents[sent_i - 1].endswith(" v."):
                # Merge the spans i and i+1 if the previous sentence ends with " v."
                # This is a hack to deal with bad tokenizations of the form Moriarty v. Holmes
                spans[-1][1] = i + len(sent)
            else:
                spans.append([i, i + len(sent)])
            curr_start = i + len(sent)
        return spans

    def span_tokenize(self, text: str):
        sents = blingfire.text_to_sentences(text).split("\n")
        spans = self.get_spans_from_sents(text, sents)
        return spans

    def tokenize(self, text: str):
        spans = self.span_tokenize(text)
        return apply_spans(text, spans)


class SemicolonTokenizer:
    def __init__(self, min_words: int = 5):
        self.min_words = min_words

    def span_tokenize(self, text: str):
        """
        Split text at semicolons if it exceeds the threshold length and return the spans.

        Args:
            text (str): Input text.
            span (tuple[int, int]): Start and end indices of the span.
            min_words (int): Minimum words per split. Skips split if the span is too short. Default: 6.
            threshold (int): Maximum words before splitting. Default: 30.

        Returns:
            list[tuple[int, int]]: List of split spans.
        """
        spans = []
        cur_start = 0
        in_quote = False
        quote_char = None
        for i in range(len(text)):
            if text[i] in ['"', "'"]:
                if not in_quote:
                    in_quote = True
                    quote_char = text[i]
                elif text[i] == quote_char:
                    in_quote = False
                    quote_char = None
            elif (
                text[i] == ";"
                and not in_quote
                and len(text[cur_start:i].split()) >= self.min_words
            ):
                spans.append((cur_start, i + 1))
                cur_start = i + 1
        if cur_start < len(text):
            spans.append((cur_start, len(text)))
        return spans


bling_tokenizer = BlingSentTokenizer()
semicolon_tokenizer = SemicolonTokenizer(min_words=5)


def merge_spans_by_imbalanced_quotes(
    text: str,
    tokenizations: list,
    verbose: bool = False,
    max_tokens: int = 45,
    error_on_unclosed_quotes: bool = True,
):
    current_start_quote = None
    merged_tokenizations = []
    if verbose:
        for i, (s, e) in enumerate(tokenizations):
            print(f"{i}: {text[s:e]}")
    is_orig_text_quote_unbalanced = find_any_unbalanced_start_quote(text)
    for s_new, e_new in tokenizations:
        if verbose:
            print(f"\n Processing span: {s_new}:{e_new}")
        if not current_start_quote:
            # No pending start quote to close
            current_start_quote = find_any_unbalanced_start_quote(text[s_new:e_new])
            if current_start_quote and verbose:
                print(f"Found {current_start_quote} at {s_new}: {text[s_new:e_new]}")
            merged_tokenizations.append((s_new, e_new))
            continue

        s_top, e_top = merged_tokenizations[-1]
        end_quote = _QUOTES[current_start_quote]
        if verbose:
            print(
                f"We are within a quote {current_start_quote}. Current span: {text[s_top:e_new]}"
            )
        # Merging the new span with the previous one exceeds the max_tokens limit.
        if len(text[s_top:e_new].split()) > max_tokens:
            if verbose:
                print(
                    "Exceeded max_tokens limit. Doing force merge. Recomputing current_start_quote."
                )
            merged_tokenizations.append((s_new, e_new))
            current_start_quote = find_any_unbalanced_start_quote(text[s_new:e_new])
            continue

        # Found the end quote in the new span.
        if end_quote in text[s_new:e_new]:
            if verbose:
                end_quote_index = text[s_new:e_new].find(end_quote)
                print(
                    f"Found {end_quote} for {current_start_quote} at {end_quote_index}."
                    " Setting current_start_quote to None."
                )
            current_start_quote = None
        # Merge the current span with the previous one
        merged_tokenizations[-1] = s_top, e_new

    if (
        not is_orig_text_quote_unbalanced
        and current_start_quote
        and error_on_unclosed_quotes
    ):
        print("Unclosed quotes detected:")
        for i, (s, e) in enumerate(merged_tokenizations):
            print(f"{i}: {text[s:e]}")
        s, e = merged_tokenizations[-1]
        raise RuntimeError(
            f"Error while processing question clue: \n{text[s:e]}: Unclosed quotes "
            f"'{current_start_quote}'."
        )

    return merged_tokenizations


def merge_spans_by_case_min_words(
    text: str,
    spans: list[tuple[int, int]],
    min_words: int = 5,
    max_words: int = 40,
    verbose: bool = False,
):
    merged_spans = [spans[0]]
    for i in range(1, len(spans)):
        curr_start, curr_end = merged_spans[-1]
        new_start, new_end = spans[i]
        curr_n_tokens = len(text[curr_start:curr_end].split())
        merged_n_tokens = len(text[curr_start:new_end].split())

        if (
            curr_n_tokens <= min_words
            or text[new_start] == "/"
            or (text[new_start].islower() and merged_n_tokens <= max_words)
        ):
            if verbose:
                print(
                    f"Undesired split detected, merging the spans {(curr_start, curr_end)} and {(new_start, new_end)}."
                )
                print(f"Current clue: {text[curr_start:curr_end]}")
                print(f"New clue    : {text[new_start:new_end]}")
            merged_spans[-1] = curr_start, new_end
        else:
            merged_spans.append((new_start, new_end))
    return merged_spans


def span_tokenize_and_merge_correct(
    tokenizer: PunktSentenceTokenizer | BlingSentTokenizer,
    text: str,
    merge_correct: bool = True,
    min_words: int = 5,
    offset: int = 0,
    raise_merge_error: bool = True,
    verbose: bool = False,
):
    spans = list(tokenizer.span_tokenize(text))
    if merge_correct:
        spans = merge_spans_by_case_min_words(
            text, spans, min_words=min_words, verbose=verbose
        )
        spans = merge_spans_by_imbalanced_quotes(
            text,
            spans,
            verbose=verbose,
            error_on_unclosed_quotes=raise_merge_error,
        )
    spans = [(s + offset, e + offset) for s, e in spans]
    return spans


def tokenize_long_sentences(
    tokenizer: PunktSentenceTokenizer | BlingSentTokenizer | SemicolonTokenizer,
    text: str,
    spans: list[tuple[int, int]],
    token_threshold: int = 40,
    merge_correct: bool = True,
    verbose: bool = False,
):
    new_spans = []
    for start, end in spans:
        sent = text[start:end]
        if len(sent.split()) < token_threshold:
            new_spans.append((start, end))
            continue
        split_spans = span_tokenize_and_merge_correct(
            tokenizer,
            sent,
            merge_correct,
            offset=start,
            verbose=verbose,
            raise_merge_error=False,
        )
        new_spans.extend(split_spans)
    return new_spans


def generate_punkt_sent_spans(
    text: str,
    verbose: bool = False,
    return_sents=False,
    min_words=5,
):
    spans = span_tokenize_and_merge_correct(punkt_sent_tokenizer, text, verbose=verbose)
    spans = tokenize_long_sentences(
        bling_tokenizer, text, spans, merge_correct=True, verbose=verbose
    )
    spans = tokenize_long_sentences(
        semicolon_tokenizer, text, spans, merge_correct=False, verbose=verbose
    )
    if return_sents:
        sents = [text[s:e] for s, e in spans]
        return spans, sents
    return [tuple(t) for t in spans]


def generate_blingfire_spans(text: str, min_words: int = 5, verbose: bool = False):
    spans = bling_tokenizer.span_tokenize(text)
    spans = merge_spans_by_case_min_words(
        text, spans, min_words=min_words, verbose=verbose
    )
    spans = merge_spans_by_imbalanced_quotes(text, spans, verbose=verbose)
    spans = tokenize_long_sentences(
        semicolon_tokenizer, text, spans, merge_correct=False, verbose=verbose
    )
    return spans


def select_span_by_size_dist(spans1, spans2):
    spans1_min = min((x[1] - x[0] for x in spans1))
    spans2_min = min((x[1] - x[0] for x in spans2))
    spans1_max = max((x[1] - x[0] for x in spans1))
    spans2_max = max((x[1] - x[0] for x in spans2))
    if spans1_max > 1.5 * spans2_max or spans1_min < 0.3 * spans2_min:
        return spans2
    return spans1


def get_clue_spans(
    qb_dict: Mapping | str, tokenization_scheme: str = "best", verbose: bool = False
):
    if isinstance(qb_dict, str):
        q = qb_dict
        qanta_spans = None
    else:
        q = qb_dict["text"]
        qanta_spans = qb_dict.get("tokenizations", None)

    if tokenization_scheme == "qanta":
        if qanta_spans:
            return qanta_spans
        raise RuntimeError(
            "Qanta Tokenizations not found in question dictionary. Please provide a valid question dictionary, or use a different tokenization scheme."
        )
    q = q.replace("“", '"').replace("”", '"')
    bf_spans = generate_blingfire_spans(q, min_words=5, verbose=verbose)
    punkt_spans = generate_punkt_sent_spans(q, min_words=5, verbose=verbose)

    if tokenization_scheme == "punkt":
        return punkt_spans
    elif tokenization_scheme == "blingfire":
        return bf_spans
    elif tokenization_scheme == "best":
        # Check the largest and smallest span
        return select_span_by_size_dist(bf_spans, punkt_spans)
    else:
        raise ValueError(f"Invalid tokenization_scheme '{tokenization_scheme}'.")


def get_clues(qb_dict: Mapping, tokenization_scheme: str = "best"):
    spans = get_clue_spans(qb_dict, tokenization_scheme)
    return apply_spans(qb_dict["text"], spans)


# Test the function with some examples
if __name__ == "__main__":
    test_text = "This is a test; it has multiple parts; each part is separated by semicolons; let's see how it works."
    test_text2 = "This is a long full sentence without semicolons; let's see how it works when the sentence is too long."
    for text in [test_text, test_text2]:
        result = tokenize_long_sentences(
            semicolon_tokenizer,
            text,
            [(0, len(text))],
            token_threshold=10,
            merge_correct=False,
        )
        print(f"Input: {text}")
        print(f"Result: {result}")
        print(f"Split sentences: {[text[s:e] for s, e in result]}")

    text = """This is the first sentence, 
    the hardest clue.
    The sentence after the first clue 
    is the second clue. For 10 points, name this 
    entity for which this is a giveaway clue."""

    qb_dict = {"text": text}
    for i, clue in enumerate(get_clues(qb_dict)):
        print(i + 1, repr(clue))

    text = '''A philosopher from this country who popularized the phrase "living in
    truth" opened an essay riffing on Marx in saying the specter haunting
    Europe was "what in the West is called ‘dissent.’" The "solidarity of
    the shaken" advocated by an author from this country inspired a
    countrymate who described the apathy of a  grocer to a sign reading
    "workers of the world, unite!"'''
    print(list(punkt_sent_tokenizer.span_tokenize(text)))
    print(generate_blingfire_spans(text))
    print(get_clue_spans(text, tokenization_scheme="best"))
    print()

    text = """  Two musicians’ parts start one bar apart and gradually merge in a
    canon from this piece that starts with the quarter notes "B, C-sharp,
    A, G-sharp, F-sharp, E." Reminders to play molto dolce, sempre dolce,
    and dolcissimo intensify under a rocking motif in this piece that is
    first stated as a broken dominant  ninth chord and is developed in
    this piece’s Recitativo-Fantasia."""
    print(list(punkt_sent_tokenizer.span_tokenize(text)))
    print(generate_blingfire_spans(text))
    print(get_clue_spans(text, tokenization_scheme="best"))
    print()

    text = """  In a poem by an author with this surname, the speaker says "You are of
  elm-shaded streets with little shops where they sell kites and
  marbles" to objects that call "‘Goose-quill men, goose-quill men, /
  May is a month for flitting.’" That author with this surname included
  a poem where the speaker repeats "False blue, / White, / Purple"
  titled "Lilacs" in the collection What’s O’Clock."""
    print(list(punkt_sent_tokenizer.span_tokenize(text)))
    print(generate_blingfire_spans(text))
    print(get_clue_spans(text, tokenization_scheme="best"))
    print()

    # get_clue_spans(text, tokenization_scheme="best")
    # generate_blingfire_spans(text)
    # %%
