def insert_team(connection, team):
    with connection.cursor() as cursor:
        # Create a new Team record
        sql = "INSERT INTO teams (`name`, `code`) VALUES (%s, %s)"
        cursor.execute(sql, (team.name, team.code))
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()


def insert_teams(connection, teams):
    for team in teams:
        insert_team(connection, team)


def insert_season(connection, season):
    with connection.cursor() as cursor:
        # Create a new Season record
        sql = "INSERT INTO seasons (`name`, short) VALUES (%s, %s)"
        cursor.execute(sql, (season.full, season.short))
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()


def insert_seasons(connection, seasons):
    for season in seasons:
        insert_season(connection, season)


def insert_player(connection, player):
    with connection.cursor() as cursor:
        # Create a new Player record
        sql = "INSERT INTO players(`name`, `lastname`, `birthdate`, `slug`) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (player.name, player.lastname, player.birthdate.strftime('%Y-%m-%d'), player.slug))
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()


def insert_players(connection, players):
    for player in players:
        insert_player(connection, player)


def insert_player_season_pair(connection, player_season):
    with connection.cursor() as cursor:
        # Create a new Player_Season record
        sql = "INSERT INTO player_season(p_id, s_id, t_id, age) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (player_season.p_id, player_season.s_id, player_season.t_id, player_season.age))
        # connection is not autocommit by default. So you must commit to save
        # your changes.
    connection.commit()


def insert_player_season_pairs(connection, player_season_list):
    for player_season in player_season_list:
        insert_player_season_pair(connection, player_season)


def insert_game(connection, season_id, team_id, opponent_id,  game):
    with connection.cursor() as cursor:
        # Create a new Game record
        sql = """INSERT INTO games(season_id, game_in_season, `date`, team_id, opponent_id, team_HA, team_points, opponent_points, team_W, team_L, team_streak) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, (season_id, game[0], game[1], team_id, opponent_id, game[5], game[9], game[10], game[11], game[12], game[13]))
        # connection is not autocommit by default. So you must commit to save
        # your changes.
    connection.commit()


def insert_player_game(connection, game):
    with connection.cursor() as cursor:
        # Create a new Player_Game record
        sql = """INSERT INTO player_game(team_game, date, ha, opponent_id, pl_season_id, player_game, 
                    last_game_data,game_started, minutes_played, fg, fga, fg_pct, tp, tpa, tp_pct, ft, fta, ft_pct, orb,
                    drb, ast, stl, blk, tov, pf, pts_last_game, game_score_index, plus_minus, team_win_pct, team_streak, 
                    opponent_win_pct, opponent_streak, opponent_streak_in_lg, pts, pts_margin, under_odd, over_odd, pts_result)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, (game[0], game[1], game[2], game[3], game[4], game[5], game[6], game[7], game[8], game[9],
                             game[10], game[11], game[12], game[13], game[14], game[15], game[16], game[17], game[18],
                             game[19], game[20], game[21], game[22], game[23], game[24], game[25], game[26], game[27],
                             game[28], game[29], game[30], game[31], game[32], game[33], game[34], game[35], game[36], game[37]))
        # connection is not autocommit by default. So you must commit to save
        # your changes.
    connection.commit()


# check how many games that team played in given season are recorded in db
def number_of_games_for_team_in_given_season(connection, season_id, team_id):
    with connection.cursor() as cursor:
        sql = "SELECT COUNT(*) FROM games g WHERE g.team_id=%s and g.season_id=%s"
        cursor.execute(sql, (team_id, season_id))
        result = cursor.fetchone()
    return result[0]


def get_season_by_short_representation(connection, season_short):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM seasons s WHERE s.short=%s"
        cursor.execute(sql, (season_short))
        result = cursor.fetchone()
    return result


def get_seasons_by_short_representation(connection, seasons_short):
    result = [get_season_by_short_representation(connection, season) for season in seasons_short]
    return result


def get_team_id_by_code(connection, code):
    with connection.cursor() as cursor:
        sql = "SELECT t.id FROM teams t WHERE t.code=%s"
        cursor.execute(sql, (code))
        result = cursor.fetchone()
    return result[0]


def get_team_win_pct_and_streak_after_game(connection, team_game, team_id, season_id):
    with connection.cursor() as cursor:
        sql = "SELECT g.team_W, g.team_streak FROM games g WHERE g.game_in_season=%s and g.team_id=%s and g.season_id=%s"
        cursor.execute(sql, (team_game, team_id, season_id))
        result = cursor.fetchone()
    win_pct = round(result[0] / team_game, 3)
    return (win_pct, result[1])


def get_player_season(connection, player_id, season_id):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM player_season ps WHERE ps.p_id=%s and ps.s_id=%s"
        cursor.execute(sql, (player_id, season_id))
        result = cursor.fetchone()
        # result = cursor.fetchall()
    return result


def get_player_id_by_slug(connection, slug):
    with connection.cursor() as cursor:
        sql = "SELECT p.id FROM players p WHERE p.slug=%s"
        cursor.execute(sql, (slug))
        result = cursor.fetchone()
    return result[0]


def find_game(connection, team_id, opponent_id, date):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM games g WHERE g.team_id=%s and g.opponent_id=%s and g.date=%s"
        cursor.execute(sql, (team_id, opponent_id, date))
        result = cursor.fetchone()
    return result


def get_player_game(connection, team_game, player_season_id):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM player_game pg WHERE pg.team_game=%s and pg.pl_season_id=%s"
        cursor.execute(sql, (team_game, player_season_id))
        result = cursor.fetchone()
    return result


def get_player_season_id(connection, player_id, season_id, team_code):
    with connection.cursor() as cursor:
        sql = """SELECT *
                  FROM player_season ps
                  WHERE ps.p_id = %s and ps.s_id = %s
                  and ps.t_id = (SELECT t.id FROM teams t WHERE t.code = %s)"""
        cursor.execute(sql, (player_id, season_id, team_code))
        result = cursor.fetchone()
    return result

