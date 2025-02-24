GAME_INFO_QUERY = """
SELECT
    game.id as id,
    t1.slug as team1,
    t2.slug as team2,
    t.slug as tournament,
    t.level as level,
    r.number as round,
    p.name as packet,
    qs.slug as qset,
    qse.slug as edition,
    qs.difficulty as difficulty
FROM
    game
JOIN
    round r ON game.round_id = r.id
JOIN
    tournament t ON r.tournament_id = t.id
JOIN
    packet p ON r.packet_id = p.id
JOIN
    question_set_edition qse ON p.question_set_edition_id = qse.id
JOIN
    question_set qs ON qse.question_set_id = qs.id
LEFT JOIN
    team t1 ON game.team_one_id = t1.id
LEFT JOIN
    team t2 ON game.team_two_id = t2.id
"""

PLAYER_INFO_QUERY = """
SELECT
    player.id as id,
    player.slug as slug,
    player.name as name,
    team.slug as team,
    tournament.slug as tournament
FROM
    player
JOIN
    team ON player.team_id = team.id
JOIN
    tournament ON team.tournament_id = tournament.id;"""

TOSSUP_INFO_QUERY = """
SELECT
    tu.id as id,
    tu.question_id as question_id,
    q.slug as slug,
    tu.question as question,
    tu.answer as answer,
    tu.answer_sanitized as answer_sanitized,
    tu.answer_primary as answer_primary,
    q.category_slug as category,
    q.subcategory_slug as subcategory,
    q.category_main_slug as category_main
FROM
    tossup tu
JOIN
    question q ON tu.question_id = q.id
"""

BONUS_PART_INFO_QUERY = """
SELECT
    bp.id as id,
    bp.bonus_id as bonus_id,
    b.question_id as question_id,
    q.slug as slug,
    b.leadin as leadin,
    b.leadin_sanitized as leadin_sanitized,
    bp.part as part,
    bp.part_number as part_number,
    bp.part_sanitized as part_sanitized,
    bp.answer as answer,
    bp.answer_sanitized as answer_sanitized,
    bp.answer_primary as answer_primary,
    bp.value as value,
    bp.difficulty_modifier as difficulty_modifier,
    q.category_slug as category,
    q.subcategory_slug as subcategory,
    q.category_main_slug as category_main
FROM
    bonus_part bp
JOIN
    bonus b ON bp.bonus_id = b.id
JOIN
    question q ON b.question_id = q.id
"""

BUZZPOINTS_INFO_QUERY = """SELECT
    b.id AS id,
    b.player_id AS player_id,
    b.tossup_id AS tossup_id,
    b.game_id AS game_id,
    tu.question AS question_text,
    tu.answer AS question_answer,
    q.category AS question_category,
    q.subcategory AS question_subcategory,
    tou.slug AS tournament,
    t.slug AS team,
    p.slug AS player,
    b.buzz_position AS buzz_position,
    b.value AS value
FROM
    buzz b
JOIN
    player p ON b.player_id = p.id
JOIN
    team t ON p.team_id = t.id
JOIN
    tossup tu ON b.tossup_id = tu.id
JOIN
    question q ON tu.question_id = q.id
JOIN
    game g ON b.game_id = g.id
JOIN
    round r ON g.round_id = r.id
JOIN
    tournament tou ON r.tournament_id = tou.id
"""
