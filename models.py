from sqlalchemy import (
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


def validate_packet_question(mapper, connection, target):
    session = Session.object_session(target)
    question = session.query(Question).filter_by(id=target.question_id).one()
    packet = session.query(Packet).filter_by(id=target.packet_id).one()

    if question.question_set_edition_id != packet.question_set_edition_id:
        raise ValueError(
            f"Question {question.id} is associated with a different question_set_edition "
            f"than Packet {packet.id}"
        )


def to_dict(obj):
    excl = ("_sa_adapter", "_sa_instance_state")
    return {
        k: v
        for k, v in vars(obj).items()
        if not k.startswith("_") and not any(hasattr(v, a) for a in excl)
    }


class Base(DeclarativeBase):
    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({params})"

    def to_dict(self):
        return to_dict(self)

    def unique_key(self):
        unique_constraints = [
            c for c in self.__table__.constraints if isinstance(c, UniqueConstraint)
        ]
        if len(unique_constraints) == 0:
            return None
        uk_columns = unique_constraints[0].columns
        return ", ".join([str(getattr(self, c.key)) for c in uk_columns])

    @classmethod
    def non_pk_columns(cls):
        return [c for c in cls.__table__.columns if not c.primary_key]


class QuestionSet(Base):
    __tablename__ = "question_set"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    slug = Column(String)
    difficulty = Column(String)

    editions = relationship("QuestionSetEdition", back_populates="question_set")

    __table_args__ = (UniqueConstraint("slug", name="uq_question_set_slug"),)


class QuestionSetEdition(Base):
    __tablename__ = "question_set_edition"

    id = Column(Integer, primary_key=True)
    question_set_id = Column(Integer, ForeignKey("question_set.id"))
    name = Column(String)
    slug = Column(String)
    date = Column(Date)

    question_set = relationship("QuestionSet", back_populates="editions")
    questions = relationship("Question", back_populates="question_set_edition")
    packets = relationship("Packet", back_populates="question_set_edition")
    tournaments = relationship("Tournament", back_populates="question_set_edition")

    __table_args__ = (
        UniqueConstraint("question_set_id", "slug", name="uq_qset_edition_set_id_slug"),
    )

    @property
    def full_slug(self):
        return f"{self.question_set.slug}_{self.slug}"


class Packet(Base):
    __tablename__ = "packet"

    id = Column(Integer, primary_key=True)
    question_set_edition_id = Column(Integer, ForeignKey("question_set_edition.id"))
    name = Column(String)

    question_set_edition = relationship("QuestionSetEdition", back_populates="packets")
    packet_questions = relationship("PacketQuestion", back_populates="packet")
    rounds = relationship("Round", back_populates="packet")

    __table_args__ = (UniqueConstraint("name", name="uq_packet_name"),)


class PacketQuestion(Base):
    __tablename__ = "packet_question"

    id = Column(Integer, primary_key=True)
    packet_id = Column(Integer, ForeignKey("packet.id"))
    question_number = Column(Integer)
    question_id = Column(Integer, ForeignKey("question.id"))

    packet = relationship("Packet", back_populates="packet_questions")
    question = relationship("Question", back_populates="packet_questions")

    __table_args__ = (
        UniqueConstraint(
            "packet_id",
            "question_id",
            name="uq_packet_question_packet_id_question_number",
        ),
    )


class Question(Base):
    __tablename__ = "question"

    id = Column(Integer, primary_key=True)
    slug = Column(String)
    question_metadata = Column("metadata", String, key="question_metadata")
    author = Column(String)
    editor = Column(String)
    category = Column(String)
    category_slug = Column(String)
    subcategory = Column(String)
    subcategory_slug = Column(String)
    subsubcategory = Column(String)
    category_main = Column(String)
    category_main_slug = Column(String)
    category_full = Column(String)

    # New column added for consistency. Maybe not be present.
    # Run `python add_missing_columns.py` to add it to the database.
    question_set_edition_id = Column(
        Integer, ForeignKey("question_set_edition.id"), nullable=True
    )

    packet_questions = relationship("PacketQuestion", back_populates="question")
    tossups = relationship("Tossup", back_populates="question")
    question_set_edition = relationship(
        "QuestionSetEdition", back_populates="questions"
    )

    __table_args__ = (
        UniqueConstraint(
            "slug",
            "author",
            "editor",
            "question_metadata",
            "question_set_edition_id",
            name="uq_question_slug_metadata",
        ),
    )

    def has_tossups(self):
        return len(self.tossups) > 0


class Tossup(Base):
    __tablename__ = "tossup"

    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("question.id"))
    question_text = Column("question", String, key="question_text")
    answer = Column(String)
    answer_sanitized = Column(String)
    answer_primary = Column(String)

    question = relationship("Question", back_populates="tossups")
    buzzes = relationship("Buzz", back_populates="tossup")

    __table_args__ = (
        UniqueConstraint(
            "question_id",
            "question_text",
            name="uq_tossup_question_id_text",
        ),
    )


class Tournament(Base):
    __tablename__ = "tournament"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    slug = Column(String)
    question_set_edition_id = Column(Integer, ForeignKey("question_set_edition.id"))
    location = Column(String)
    level = Column(String)
    start_date = Column(Date)
    end_date = Column(Date)

    question_set_edition = relationship(
        "QuestionSetEdition", back_populates="tournaments"
    )
    rounds = relationship("Round", back_populates="tournament")
    teams = relationship("Team", back_populates="tournament")

    __table_args__ = (UniqueConstraint("slug", name="uq_tournament_slug"),)


class Round(Base):
    __tablename__ = "round"

    id = Column(Integer, primary_key=True)
    tournament_id = Column(Integer, ForeignKey("tournament.id"))
    number = Column(Integer)
    packet_id = Column(Integer, ForeignKey("packet.id"))
    exclude_from_individual = Column(Boolean)

    tournament = relationship("Tournament", back_populates="rounds")
    packet = relationship("Packet", back_populates="rounds")
    games = relationship("Game", back_populates="round")


class Team(Base):
    __tablename__ = "team"

    id = Column(Integer, primary_key=True)
    tournament_id = Column(Integer, ForeignKey("tournament.id"))
    name = Column(String)
    slug = Column(String)

    tournament = relationship("Tournament", back_populates="teams")
    players = relationship("Player", back_populates="team")
    games_as_team_one = relationship(
        "Game", foreign_keys="[Game.team_one_id]", back_populates="team_one"
    )
    games_as_team_two = relationship(
        "Game", foreign_keys="[Game.team_two_id]", back_populates="team_two"
    )

    __table_args__ = (
        UniqueConstraint("tournament_id", "slug", name="uq_team_tournament_slug"),
    )


class Player(Base):
    __tablename__ = "player"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("team.id"))
    name = Column(String)
    slug = Column(String)

    team = relationship("Team", back_populates="players")
    buzzes = relationship("Buzz", back_populates="player")

    __table_args__ = (UniqueConstraint("team_id", "slug", name="uq_player_team_slug"),)


class Game(Base):
    __tablename__ = "game"

    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey("round.id"))
    tossups_read = Column(Integer)
    team_one_id = Column(Integer, ForeignKey("team.id"))
    team_two_id = Column(Integer, ForeignKey("team.id"))

    round = relationship("Round", back_populates="games")
    team_one = relationship(
        "Team", foreign_keys=[team_one_id], back_populates="games_as_team_one"
    )
    team_two = relationship(
        "Team", foreign_keys=[team_two_id], back_populates="games_as_team_two"
    )
    buzzes = relationship("Buzz", back_populates="game")


class Buzz(Base):
    __tablename__ = "buzz"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("player.id"))
    game_id = Column(Integer, ForeignKey("game.id"))
    tossup_id = Column(Integer, ForeignKey("tossup.id"))
    buzz_position = Column(Integer)
    value = Column(Integer)

    player = relationship("Player", back_populates="buzzes")
    game = relationship("Game", back_populates="buzzes")
    tossup = relationship("Tossup", back_populates="buzzes")

    __table_args__ = (
        UniqueConstraint(
            "player_id",
            "game_id",
            "tossup_id",
            "buzz_position",
            name="uq_player_game_tossup_buzz_position",
        ),
    )


event.listen(PacketQuestion, "before_insert", validate_packet_question)
event.listen(PacketQuestion, "before_update", validate_packet_question)


all_classes = [
    QuestionSet,
    QuestionSetEdition,
    Packet,
    PacketQuestion,
    Question,
    Tossup,
    Tournament,
    Round,
    Team,
    Player,
    Game,
    Buzz,
]


def create_session(db_path, create_tables=False):
    engine = create_engine(f"sqlite:///{db_path}")
    if create_tables:
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def query_buzzes(session, filters=None, limit=None):
    """
    Query buzzes from the database with optional filters.

    :param session: SQLAlchemy session
    :param filters: Dictionary of filters to apply to the query
    :param limit: Maximum number of results to return
    :return: List of Buzz objects
    """
    query = session.query(Buzz)

    if filters:
        for key, value in filters.items():
            if hasattr(Buzz, key):
                query = query.filter(getattr(Buzz, key) == value)

    if limit:
        query = query.limit(limit)

    return query.all()


if __name__ == "__main__":
    session = create_session("data/sst-23-24-cleaned.db")

    # Query buzzes
    filters = {"player_id": 1, "value": 10}
    limit = 5
    buzzes = query_buzzes(session, filters=filters, limit=limit)
    for buzz in buzzes:
        print(buzz)

# %%
