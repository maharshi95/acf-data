import msgspec

import models
from utils import acf_sanitization, qb_tokenization


class QuestionMetadata(msgspec.Struct):
    category: str
    subcategory: list[str]
    category_main: str
    category_full: str
    difficulty: str
    question_set: str


class QuizbowlQuestion(msgspec.Struct):
    qid: str
    question: str
    answer: str
    answer_primary: str
    clue_spans: list[tuple[int, int]]
    metadata: QuestionMetadata

    def clues(self):
        clues = []
        for span in self.clue_spans:
            clues.append(self.question[span[0] : span[1]].strip())
        return clues


def create_tossup_entry(tossup: models.Tossup):
    question_sanitized = acf_sanitization.sanitize_question(tossup.question_text)
    clue_spans = qb_tokenization.get_clue_spans(
        question_sanitized, tokenization_scheme="best"
    )
    question = tossup.question
    qset = question.question_set_edition.question_set
    return QuizbowlQuestion(
        qid=f"acf-{tossup.id}",
        answer=tossup.answer_sanitized,
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
        ),
    )
