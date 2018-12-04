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
        sql = "INSERT INTO players(`name`, `lastname`, `birthdate`) VALUES (%s, %s, %s)"
        cursor.execute(sql, (player.name, player.lastname, player.birthdate.strftime('%Y-%m-%d')))
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
        sql = "INSERT INTO games(season_id, game_in_season, `date`, team_id, opponent_id, team_HA, team_points, opponent_points, team_W, team_L, team_streak) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (season_id, game[0], game[1], team_id, opponent_id, game[5], game[9], game[10], game[11], game[12], game[13]))
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

