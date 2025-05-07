def prepare_token_indices(
    question_text: str, clue_spans: list[tuple[int, int]], run_length: int = 7
) -> tuple[list[int], list[int]]:
    """
    Prepares token indices for tossup questions by converting character-based clue spans
    to token-based indices and generating run indices at regular intervals.

    This function is used to prepare token indices for the QuizbowlQuestion structure,
    enabling the creation of question runs for training and evaluation purposes.

    Args:
        question_text: The sanitized question text.
        clue_spans: A list of (start, end) character position tuples marking clue boundaries.
        run_length: The number of tokens between each run index, defaults to 7.

    Returns a tuple of two lists:
        clue_indices: List of token indices corresponding to the end of each clue.
        run_indices: List of token indices at regular intervals (every run_length tokens)
            and at clue boundaries, used for generating question runs.
    """

    # Convert character-based clue spans to token-based indices
    clue_indices = []
    for start, end in clue_spans:
        tokens = question_text[:end].split()
        clue_indices.append(len(tokens) - 1)

    # Generate run indices at regular intervals
    run_indices = [run_length - 1]
    ptr = 0
    while ptr < len(clue_indices):
        new_index = run_indices[-1] + run_length
        if new_index < clue_indices[ptr]:
            run_indices.append(new_index)
        else:
            run_indices.append(clue_indices[ptr])
            ptr += 1
    return clue_indices, run_indices
