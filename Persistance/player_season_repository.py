from Services.db_connection import connection


def insert(player_season):
    with connection.cursor() as cursor:
        # Create a new Player_Season record
        sql = "INSERT INTO player_season (p_id, s_id, t_id, age) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (player_season.p_id, player_season.s_id, player_season.t_id, player_season.age))
        # connection is not autocommit by default. So you must commit to save
        # your changes.
    connection.commit()


def insert_multiple_player_season_rows(player_season_list):
    for player_season in player_season_list:
        insert(player_season)


# def get_player_season_id(player_id, season_id, team_code):
def get_player_season_id(player_id, season_id, team_code):
    with connection.cursor() as cursor:
        sql = """SELECT *
                  FROM player_season ps
                  WHERE ps.p_id = %s and ps.s_id = %s
                  and ps.t_id = (SELECT t.id FROM teams t WHERE t.code = %s)"""
        cursor.execute(sql, (player_id, season_id, team_code))
        result = cursor.fetchone()
    return result


def get_player_season_id_by_player_and_season(player_id, season_id):
    with connection.cursor() as cursor:
        sql = """SELECT *
                  FROM player_season ps
                  WHERE ps.p_id = %s and ps.s_id = %s"""
        cursor.execute(sql, (player_id, season_id))
        result = cursor.fetchall()
    return result


def get_player_season_for_given_season(player_id, season_id):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM player_season ps WHERE ps.p_id=%s and ps.s_id=%s"
        cursor.execute(sql, (player_id, season_id))
        result = cursor.fetchone()
        # result = cursor.fetchall()
    return result


def get_player_season(player_id, season_id):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM player_season ps WHERE ps.p_id=%s and ps.s_id=%s"
        cursor.execute(sql, (player_id, season_id))
        result = cursor.fetchone()
        # result = cursor.fetchall()
    return result
