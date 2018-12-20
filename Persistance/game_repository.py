from Services.db_connection import connection


def insert(season_id, team_id, opponent_id,  game):
    sql_params = (season_id, game[0], game[1], team_id, opponent_id,
                  game[5], game[9], game[10], game[11], game[12], game[13])
    with connection.cursor() as cursor:
        # Create a new Game record
        sql = """INSERT INTO games (season_id, game_in_season, `date`, team_id, opponent_id, team_HA, 
                                   team_points, opponent_points, team_W, team_L, team_streak) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, sql_params)
        # connection is not autocommit by default. So you must commit to save
        # your changes.
    connection.commit()


# check how many games that team played in given season are recorded in db
def number_of_games_for_team_in_given_season(season_id, team_id):
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) FROM games g WHERE g.team_id=%s and g.season_id=%s"
        cursor.execute(sql, (team_id, season_id))
        result = cursor.fetchone()
    return result[0]


def get_team_win_pct_and_streak_after_game(team_game, team_id, season_id):
    with connection.cursor() as cursor:
        sql = """SELECT g.team_W, g.team_streak
                 FROM games g 
                 WHERE g.game_in_season=%s and g.team_id=%s and g.season_id=%s"""
        cursor.execute(sql, (team_game, team_id, season_id))
        result = cursor.fetchone()
    if result is not None:
        return round(result[0] / team_game, 3), result[1]
    return None, None


# def find_game(team_id, opponent_id, date):
def find_game_by_team_opponent_date(team_id, opponent_id, date):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM games g WHERE g.team_id=%s and g.opponent_id=%s and g.date=%s"
        cursor.execute(sql, (team_id, opponent_id, date))
        result = cursor.fetchone()
    return result


def count_games_in_given_season(season_id):
    with connection.cursor() as cursor:
        sql = "SELECT count(*) FROM games g WHERE g.season_id=%s"
        cursor.execute(sql, season_id)
        result = cursor.fetchone()
    return result[0]


def count_games_for_team_in_given_season(team_id, season_id):
    with connection.cursor() as cursor:
        sql = "SELECT count(*) FROM games g WHERE g.team_id=%s and g.season_id=%s"
        cursor.execute(sql, (team_id, season_id))
        result = cursor.fetchone()
    return result[0]


def get_game_number_for_team_in_season(date, team_id, opponent_id):
    with connection.cursor() as cursor:
        sql = "SELECT g.game_in_season FROM games g WHERE g.date=%s and g.team_id=%s and g.opponent_id=%s"
        cursor.execute(sql, (date, team_id, opponent_id))
        result = cursor.fetchone()
    return result[0]
