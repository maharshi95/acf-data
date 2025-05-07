from datasets import Dataset

from core import models
from core.structs import (
    ProgressiveClue,
    QuestionMetadata,
    QuizbowlQuestion,
)
from utils import acf_sanitization, qb_tokenization
from utils.tossups import prepare_token_indices


def create_tossup_entry(tossup: models.Tossup, prefix="acf"):
    question_sanitized = acf_sanitization.sanitize_question(tossup.question_text)
    clue_spans = qb_tokenization.get_clue_spans(
        question_sanitized, tokenization_scheme="blingfire"
    )
    answers = acf_sanitization.get_short_clean_answers(tossup.answer)
    question = tossup.question
    pq = question.packet_questions[0]
    qset = question.question_set_edition.question_set
    qid = f"{pq.packet_id}-{pq.question_number}"
    buzz_offset = acf_sanitization.get_buzz_offset(tossup.question_text)
    human_buzz_positions = sorted(
        [(b.buzz_position - buzz_offset, b.value) for b in tossup.buzzes]
    )
    return QuizbowlQuestion(
        qid=f"{prefix}-{qid}",
        answer=tossup.answer_sanitized,
        clean_answers=answers["clean"],
        explanation=answers["explanation"],
        answer_primary=tossup.answer_primary,
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


def create_progressive_clues(qb_question: QuizbowlQuestion):
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


def create_and_push_dataset(db_path: str, prefix: str):
    session = models.create_session(db_path)
    tossups = session.query(models.Tossup).all()
    tossups = [create_tossup_entry(t, prefix) for t in tossups]

    questions = [t.to_dict() for t in tossups]
    progressive_clues = []
    for t in tossups:
        clues = create_progressive_clues(t)
        for c in clues:
            progressive_clues.append(c.to_dict())

    questions_dataset = Dataset.from_list(questions)
    progressive_clues_dataset = Dataset.from_list(progressive_clues)
    questions_dataset.push_to_hub(
        f"mgor/{prefix}-tossups",
        config_name="questions",
        split="eval",
    )
    progressive_clues_dataset.push_to_hub(
        f"mgor/{prefix}-tossups", config_name="progressive-clues", split="eval"
    )

    def inject_token_indices(x):
        clue_indices, run_indices = prepare_token_indices(
            x["question"], x["clue_spans"], run_length=7
        )
        return {
            "clue_token_indices": clue_indices,
            "run_indices": run_indices,
        }

    tossups_dataset = questions_dataset.map(inject_token_indices)
    tossups_dataset.push_to_hub(
        f"umdclip/{prefix}-tossups",
        split="eval",
    )
    return questions_dataset, progressive_clues_dataset, tossups_dataset


# %%

create_and_push_dataset("data/dbs/regs25.db", "acf-regs25")
ds1, ds2, ds3 = create_and_push_dataset("data/dbs/co24-cleaned.db", "acf-co24")
