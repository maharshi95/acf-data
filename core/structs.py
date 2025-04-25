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
    human_buzz_positions: list[tuple[int, int]] = []


class BonusQuestion(JsonStruct):
    qid: str
    leadin: str
    parts: list[dict]
    metadata: QuestionMetadata


class BonusPart(JsonStruct):
    number: int
    part: str
    answer: str
    answer_primary: str
    clean_answers: list[str]
    explanation: str
    value: int
    difficulty_modifier: str


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


# model-id for the config-name of the huggingface dataset
# dataset-id is the split name of the huggingface dataset
# Example: load_dataset("umdclip/model-outputs", config_name="gpt-4o-aggressive", split="co24")
class ModelOutputs(JsonStruct):
    qid: str
    run_id: str  # {qid}#{run_number}
    answer_primary: str
    clean_answers: list[str]
    guess: str
    confidence: float
    buzz: bool
    explanation: str = ""


# We get this by post-processing the Model Outputs.
# Avoid iterating though the ModelOutputs dataset for preparing the leaderboard.
class ModelResults(JsonStruct):
    model_id: str
    model_name: str
    dataset_id: str

    # Metrics:
    buzz_accuracy: float
    win_rate_human: float
    win_rate_model: float
    explanation_helpfulness: float
    explanation_distraction: float

    # An overconfident model can have high helpfulness and high distraction.

    # A highly skeptic model has lower rate of answer flipping.

    # To compute the explanation metrics, we need to finalize a small set of models:
    # llama-3.1-8b-instruct
    # mistral
    # gpt-3.5-turbo
