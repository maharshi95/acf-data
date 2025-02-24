import msgspec

import core.models as models
from utils import acf_sanitization, qb_tokenization


class JsonStruct(msgspec.Struct, omit_defaults=True):
    """
    A class representing a JSON-serializable structure.

    This class extends the `msgspec.Struct` class and provides methods for converting
    between JSON dictionaries and instances of the class.

    Methods:
        from_dict(obj: dict) -> JsonStruct:
            Converts a dictionary object to an instance of the JsonStruct class.

        asdict() -> dict:
            Converts an instance of the JsonStruct class to a dictionary.

    Usage:
        json_struct = JsonStruct.from_dict(json_dict)
        json_dict = json_struct.asdict()
    """

    def __getitem__(self, item):
        if item not in self.__struct_fields__:
            raise KeyError(f"Field '{item}' not found in {self.__class__.__name}")
        return getattr(self, item)

    def get(self, item, default=None):
        if item not in self.__struct_fields__:
            return default
        return getattr(self, item)

    def __setitem__(self, key, value):
        if key not in self.__struct_fields__:
            raise KeyError(f"Field '{key}' not found in {self.__class__.__name}")
        setattr(self, key, value)

    def __iter__(self):
        for field in self.__struct_fields__:
            yield field, getattr(self, field)

    @classmethod
    def from_dict(cls, obj: dict):
        json_str = msgspec.json.encode(obj)
        return msgspec.json.decode(json_str, type=cls)

    def to_dict(self):
        json_str = msgspec.json.encode(self)
        return msgspec.json.decode(json_str)

    def asdict(self):
        return msgspec.structs.asdict(self)


class QuestionMetadata(JsonStruct):
    category: str
    subcategory: list[str]
    category_main: str
    category_full: str
    difficulty: str
    question_set: str
    packet: str


class QuizbowlQuestion(JsonStruct):
    qid: str
    question: str
    answer: str
    answer_primary: str
    clean_answers: list[str]
    explanation: str
    clue_spans: list[tuple[int, int]]
    metadata: QuestionMetadata

    def clues(self):
        clues = []
        for span in self.clue_spans:
            clues.append(self.question[span[0] : span[1]].strip())
        return clues


class ProgressiveClue(JsonStruct):
    qc_id: str
    clue_text: str
    clean_answers: list[str]
    orig_qid: str
    n_clues: int
    orig_question: str
    orig_answer_string: str
    clue_spans: list[tuple[int, int]]
    metadata: QuestionMetadata


def create_tossup_entry(tossup: models.Tossup, prefix="acf"):
    question_sanitized = acf_sanitization.sanitize_question(tossup.question_text)
    clue_spans = qb_tokenization.get_clue_spans(
        question_sanitized, tokenization_scheme="blingfire"
    )
    clean_answers, explanation = acf_sanitization.get_clean_answers(
        tossup.answer_sanitized
    )
    question = tossup.question
    pq = question.packet_questions[0]
    qset = question.question_set_edition.question_set
    qid = f"{pq.packet_id}-{pq.question_number}"
    return QuizbowlQuestion(
        qid=f"{prefix}-{qid}",
        answer=tossup.answer_sanitized,
        clean_answers=clean_answers,
        explanation=explanation,
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
