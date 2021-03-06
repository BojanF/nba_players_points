from Services.db_connection import connection


def insert(game):
    sql_params = (game[0], game[1], game[2], game[3], game[4], game[5], game[6], game[7], game[8], game[9],
                  game[10], game[11], game[12], game[13], game[14], game[15], game[16], game[17], game[18],
                  game[19], game[20], game[21], game[22], game[23], game[24], game[25], game[26], game[27],
                  game[28], game[29], game[30], game[31], game[32], game[33], game[34], game[35], game[36], game[37])
    with connection.cursor() as cursor:
        # Create a new Player_Game record
        sql = """INSERT INTO player_game (team_game, date, ha, opponent_id, pl_season_id, player_game, 
                    last_game_data,game_started, minutes_played, fg, fga, fg_pct, tp, tpa, tp_pct, ft, fta, 
                    ft_pct, orb, drb, ast, stl, blk, tov, pf, pts_last_game, game_score_index, plus_minus,
                    team_win_pct, team_streak, opponent_win_pct, opponent_streak, opponent_streak_in_lg, pts, 
                    pts_margin, under_odd, over_odd, pts_result)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, sql_params)
        # connection is not autocommit by default. So you must commit to save
        # your changes.
    connection.commit()


# def get_player_game(team_game, player_season_id):
def get_player_game_by_team_game_and_player_season(team_game, player_season_id):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM player_game pg WHERE pg.team_game=%s and pg.pl_season_id=%s"
        cursor.execute(sql, (team_game, player_season_id))
        result = cursor.fetchone()
    return result


def count_games_for_player_season(player_season_id):
    with connection.cursor() as cursor:
        sql = "SELECT count(*) FROM player_game pg WHERE pg.pl_season_id=%s"
        cursor.execute(sql, player_season_id)
        result = cursor.fetchone()
    return result[0]


def count_games_for_player_in_season(player_id, season_id):
    with connection.cursor() as cursor:
        sql = """SELECT count(*)    
                 FROM player_game pg 
                 WHERE pg.pl_season_id in (SELECT ps.id
                                           FROM player_season ps
                                           WHERE ps.p_id = %s and ps.s_id = %s)"""
        cursor.execute(sql, (player_id, season_id))
        result = cursor.fetchone()
    return result[0]


def get_player_games_in_season(player_id, season_id):
    with connection.cursor() as cursor:
        sql = """SELECT *  
                 FROM player_game pg 
                 WHERE pg.pl_season_id in (SELECT ps.id
                                           FROM player_season ps
                                           WHERE ps.p_id = %s and ps.s_id = %s)"""
        cursor.execute(sql, (player_id, season_id))
        result = cursor.fetchall()
    return result[0]


def get_player_games_data_set(date_start, date_end):
    with connection.cursor() as cursor:
        sql = """ SELECT pg.ha, pg.minutes_played, pg.fg, pg.fga, pg.fg_pct, pg.tp, pg.tpa, pg.tp_pct, pg.ft, pg.fta, 
                         pg.ft_pct, pg.orb, pg.drb, pg.ast, pg.stl, pg.blk, pg.tov, pg.pf, pg.pts_last_game,
                         pg.game_score_index, pg.plus_minus, pg.team_win_pct, pg.team_streak, pg.opponent_win_pct,
                         pg.opponent_streak, pg.opponent_streak_in_lg, pg.pts_margin, pg.under_odd, pg.over_odd, pg.pts_result
                  FROM player_game pg
                  WHERE pg.last_game_data = 1 and
                        pg.pts_result is not null and
                        pg.team_game BETWEEN 6 and 80 and
                        pg.date BETWEEN %s and %s"""
        cursor.execute(sql, (date_start, date_end))
        result = cursor.fetchall()
    return result


def get_player_data_set_games_feature_selected(date_start, date_end):
    with connection.cursor() as cursor:
        sql = """ SELECT pg.ha, pg.fg_pct, pg.tp, pg.tp_pct, pg.ft, pg.fta, pg.ft_pct, pg.blk, 
                         pg.opponent_win_pct, pg.under_odd, pg.over_odd, pg.pts_result
                  FROM player_game pg
                  WHERE pg.last_game_data = 1 and
                        pg.pts_result is not null and
                        pg.team_game BETWEEN 6 and 80 and
                        pg.date BETWEEN %s and %s"""
        cursor.execute(sql, (date_start, date_end))
        result = cursor.fetchall()
    return result


def get_data_set_for_player(player_id, date_start, date_end):
    with connection.cursor() as cursor:
        sql = """ SELECT pg.id, day(pg.date) as day, month(pg.date) as month, year(pg.date) as year, pg.opponent_id, 
                         pg.ha, pg.ha, pg.minutes_played, pg.fg, pg.fga, pg.fg_pct, pg.tp, pg.tpa, pg.tp_pct, pg.ft, pg.fta,
                         pg.ft_pct, pg.orb, pg.drb, pg.ast, pg.stl, pg.blk, pg.tov, pg.pf, pg.pts_last_game,
                         pg.game_score_index, pg.plus_minus, pg.team_win_pct, pg.team_streak, pg.opponent_win_pct,
                         pg.opponent_streak, pg.opponent_streak_in_lg, pg.pts_margin, pg.under_odd, pg.over_odd, pg.pts_result
                  FROM player_game pg
                  WHERE pg.pl_season_id in (
                        SELECT ps.id
                        FROM player_season ps, players p
                        WHERE p.id = %s and
                          p.id = ps.p_id) and
                        pg.last_game_data = 1 and
                        pg.pts_result is not null and
                        pg.team_game BETWEEN 6 and 80 and
                        pg.date BETWEEN %s and %s"""
        cursor.execute(sql, (player_id, date_start, date_end))
        result = cursor.fetchall()
    return result


def get_data_set_for_player_feature_selected(player_id, date_start, date_end):
    with connection.cursor() as cursor:
        sql = """ SELECT pg.id, day(pg.date) as day, month(pg.date) as month, year(pg.date) as year, pg.opponent_id, 
                         pg.ha, pg.ha, pg.fg_pct, pg.tp, pg.tp_pct, pg.ft, pg.fta, pg.ft_pct, pg.blk, 
                         pg.opponent_win_pct, pg.under_odd, pg.over_odd, pg.pts_result
                  FROM player_game pg
                  WHERE pg.pl_season_id in (
                        SELECT ps.id
                        FROM player_season ps, players p
                        WHERE p.id = %s and
                          p.id = ps.p_id) and
                        pg.last_game_data = 1 and
                        pg.pts_result is not null and
                        pg.team_game BETWEEN 6 and 80 and
                        pg.date BETWEEN %s and %s"""
        cursor.execute(sql, (player_id, date_start, date_end))
        result = cursor.fetchall()
    return result
