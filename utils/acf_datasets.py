# %%
import argparse

from datasets import Dataset
from huggingface_hub import whoami

from core import models
from core.structs import (
    BonusPart,
    ProgressiveClue,
    QBBonusQuestion,
    QBTossupQuestion,
    QuestionMetadata,
)
from utils import acf_sanitization, qb_tokenization


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


def create_tossup_entry(tossup: models.Tossup, prefix="acf"):
    question_sanitized = acf_sanitization.sanitize_question(tossup.question_text)
    clue_spans = qb_tokenization.get_clue_spans(
        question_sanitized, tokenization_scheme="blingfire"
    )
    answers = acf_sanitization.get_short_clean_answers(tossup.answer)
    question = tossup.question
    pq = question.packet_questions[0]
    qset = question.question_set_edition.question_set
    qid = f"t-{pq.packet_id}-{pq.question_number}"
    buzz_offset = acf_sanitization.get_buzz_offset(tossup.question_text)
    human_buzz_positions = sorted(
        [(b.buzz_position - buzz_offset, b.value) for b in tossup.buzzes]
    )
    return QBTossupQuestion(
        qid=f"{prefix}-{qid}",
        answer_line=tossup.answer,
        answer_primary=tossup.answer_primary,
        explanation=answers["explanation"],
        clean_answers=answers["clean"],
        clue_spans=clue_spans,
        question=question_sanitized,
        metadata=QuestionMetadata(
            category=question.category_slug,
            subcategory=[question.subcategory_slug],
            category_main=question.category_main_slug,
            category_full=question.category_full,
            difficulty=qset.difficulty.split()[0],
            question_set=qset.slug,
            packet=question.packet_questions[0].packet.name,
            human_buzz_positions=human_buzz_positions,
        ),
    )


def create_progressive_clues(qb_question: QBTossupQuestion):
    clues = []
    for i, span in enumerate(qb_question.clue_spans, 1):
        clue = ProgressiveClue(
            qc_id=f"{qb_question.qid}_{i}",
            clue_text=qb_question.question[span[0] : span[1]].strip(),
            clean_answers=qb_question.clean_answers,
            orig_qid=qb_question.qid,
            n_clues=i,
            orig_question=qb_question.question,
            orig_answer_string=qb_question.answer,
            clue_spans=qb_question.clue_spans,
            metadata=qb_question.metadata,
        )
        clues.append(clue)
    return clues


def create_bonus_entry(bonus: models.Bonus, prefix="acf"):
    leadin = acf_sanitization.sanitize_question(bonus.leadin)
    q_info = bonus.question
    pq = q_info.packet_questions[0]
    qset = q_info.question_set_edition.question_set
    qid = f"b-{pq.packet_id}-{pq.question_number}"
    parts = []
    for part in bonus.bonus_parts:
        part_text = acf_sanitization.sanitize_answer(part.part)
        answers = acf_sanitization.get_short_clean_answers(part.answer)
        parts.append(
            BonusPart(
                number=part.part_number,
                question=part_text,
                answer_line=part.answer,
                answer_primary=part.answer_primary,
                clean_answers=answers["clean"],
                explanation=answers["explanation"],
                value=part.value,
                difficulty_modifier=part.difficulty_modifier,
            )
        )
    return QBBonusQuestion(
        qid=f"{prefix}-{qid}",
        leadin=leadin,
        parts=parts,
        metadata=QuestionMetadata(
            category=q_info.category_slug,
            subcategory=[q_info.subcategory_slug],
            category_main=q_info.category_main_slug,
            category_full=q_info.category_full,
            difficulty=qset.difficulty.split()[0],
            question_set=qset.slug,
            packet=q_info.packet_questions[0].packet.name,
        ),
    )


def get_tossup_questions(db_path: str, prefix: str) -> list[QBTossupQuestion]:
    session = models.create_session(db_path)
    tossups = session.query(models.Tossup).all()
    tossups = [create_tossup_entry(t, prefix) for t in tossups]
    return tossups


def create_tossup_dataset(db_path: str, prefix: str, run_length: int = 7):
    tossups = get_tossup_questions(db_path, prefix)

    dataset = Dataset.from_list([t.to_dict() for t in tossups])

    def inject_token_indices(x):
        clue_indices, run_indices = prepare_token_indices(
            x["question"], x["clue_spans"], run_length=run_length
        )
        return {
            "clue_token_indices": clue_indices,
            "run_indices": run_indices,
        }

    dataset = dataset.map(inject_token_indices)
    return dataset


def create_progressive_clues_dataset(db_path: str, prefix: str):
    tossups = get_tossup_questions(db_path, prefix)
    progressive_clues = []
    for t in tossups:
        question_clues = create_progressive_clues(t)
        progressive_clues.extend(question_clues)

    return Dataset.from_list([c.to_dict() for c in progressive_clues])


def create_bonus_dataset(db_path: str, prefix: str):
    session = models.create_session(db_path)
    bonuses = session.query(models.Bonus).all()
    bonuses = [create_bonus_entry(b, prefix) for b in bonuses]

    return Dataset.from_list([b.to_dict() for b in bonuses])
