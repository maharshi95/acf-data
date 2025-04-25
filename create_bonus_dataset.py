# %%
from datasets import Dataset

from core import models
from core.structs import (
    BonusPart,
    BonusQuestion,
    QuestionMetadata,
)
from utils import acf_sanitization, qb_tokenization


def create_bonus_entry(bonus: models.Bonus, prefix="acf"):
    leadin = acf_sanitization.sanitize_question(bonus.leadin)
    question = bonus.question
    pq = question.packet_questions[0]
    qset = question.question_set_edition.question_set
    qid = f"{pq.packet_id}-{pq.question_number}"
    parts = []
    for part in bonus.bonus_parts:
        part_text = acf_sanitization.sanitize_answer(part.part)
        clean_answers, explanation = acf_sanitization.get_short_clean_answers(
            part.answer
        )
        parts.append(
            BonusPart(
                number=part.part_number,
                part=part_text,
                answer=part.answer_sanitized,
                answer_primary=part.answer_primary,
                clean_answers=clean_answers,
                explanation=explanation,
                value=part.value,
                difficulty_modifier=part.difficulty_modifier,
            )
        )
    return BonusQuestion(
        qid=f"{prefix}-{qid}",
        leadin=leadin,
        parts=parts,
        metadata=QuestionMetadata(
            category=question.category_slug,
            subcategory=[question.subcategory_slug],
            category_main=question.category_main_slug,
            category_full=question.category_full,
            difficulty=qset.difficulty.split()[0],
            question_set=qset.slug,
            packet=question.packet_questions[0].packet.name,
        ),
    )


def create_and_push_dataset(db_path: str, prefix: str):
    session = models.create_session(db_path)
    bonuses = session.query(models.Bonus).all()
    bonuses = [create_bonus_entry(b, prefix) for b in bonuses]

    questions = [b.to_dict() for b in bonuses]

    questions_dataset = Dataset.from_list(questions)
    questions_dataset.push_to_hub(
        f"mgor/{prefix}-bonuses",
        config_name="questions",
        split="eval",
    )
    return questions_dataset


# %%

ds = create_and_push_dataset("data/dbs/regs25.db", "acf-regs25")
ds = create_and_push_dataset("data/dbs/co24-cleaned.db", "acf-co24")

# %%

ds.push_to_hub("umdclip/acf-co24-bonuses", split="eval", private=True)
# %%
