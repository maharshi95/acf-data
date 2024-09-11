# %%
"""
This script checks for various inconsistencies in the database. For example, it checks if:
- A buzz is associated with the same question set edition as the game and team's tournament
- A tossup is tournaments around the same date
- Tournament levels associated with a question set are the same

Example usage:

```bash
python check_consistencies.py <db_path>
```
"""

import sys
from datetime import timedelta

import models
from models import create_session

db_path = sys.argv[1]

print(f"Checking database at {db_path}")
session = create_session(db_path)


# Check if a round always points to the same question_set_edition
for round in session.query(models.Round).all():
    assert (
        round.packet.question_set_edition_id == round.tournament.question_set_edition_id
    )

# Check if a packet question always points to the same question_set_edition
for pq in session.query(models.PacketQuestion).all():
    assert pq.question.question_set_edition_id == pq.packet.question_set_edition_id


# Check if a game from a tournament only has participants from that tournament
for game in session.query(models.Game).all():
    assert game.round.tournament_id == game.team_one.tournament_id
    assert game.round.tournament_id == game.team_two.tournament_id

# Check if a buzz is associated with the same question set edition as the game and team's tournament
for buzz in session.query(models.Buzz).all():
    assert (
        buzz.tossup.question.question_set_edition_id
        == buzz.game.round.tournament.question_set_edition_id
    )

    assert (
        buzz.tossup.question.question_set_edition_id
        == buzz.player.team.tournament.question_set_edition_id
    )


# Check if a tossup is tournaments around the same date
ALLOWED_GAP = timedelta(days=7)
for tossup in session.query(models.Tossup).all():
    tourney_dates = set()
    for buzz in tossup.buzzes:
        tourney_dates.add(buzz.game.round.tournament.end_date)
    if len(tourney_dates) == 0:
        continue
    max_gap = max(tourney_dates) - min(tourney_dates)
    if max_gap > ALLOWED_GAP:
        dates = [d.strftime("%Y-%m-%d") for d in tourney_dates]
        print(f"Tossup {tossup.id} is used in tournaments at {dates}")


# Check if tournament levels associated with a question set are the same
for qset in session.query(models.QuestionSetEdition).all():
    levels = {t.level for t in qset.tournaments}
    assert len(levels) == 1


for qset in session.query(models.QuestionSetEdition).all():
    print(qset.full_slug, qset.question_set.difficulty.split()[0])
