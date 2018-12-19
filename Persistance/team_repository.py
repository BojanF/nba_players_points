from Services.db_connection import connection


def insert(team):
    with connection.cursor() as cursor:
        # Create a new Team record
        sql = "INSERT INTO teams (`name`, `code`, `active`) VALUES (%s, %s, %s)"
        cursor.execute(sql, (team.name, team.code, team.active))
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()


def insert_teams(teams):
    for team in teams:
        insert(team)


def get_team_id_by_code(code):
    with connection.cursor() as cursor:
        sql = "SELECT t.id FROM teams t WHERE t.code=%s"
        cursor.execute(sql, code)
        result = cursor.fetchone()
    if result is not None:
        return result[0]
    return None


def get_active_teams():
    with connection.cursor() as cursor:
        sql = "SELECT * FROM teams t WHERE t.active=1"
        cursor.execute(sql)
        result = cursor.fetchall()
    return result


def get_team_by_id(team_id):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM teams t WHERE t.id=%s"
        cursor.execute(sql, team_id)
        result = cursor.fetchone()
    return result


def set_team_inactive(team_id):
    team_to_update = get_team_by_id(team_id)
    if team_to_update is None:
        print('Team with id ' + str(team_id) + ' does not exist')
        return
    new_code = team_to_update[2] + '_' + str(team_to_update[0])
    update_team(team_id, team_to_update[1], new_code, 0)


def update_team(team_id, name, code, active):
    with connection.cursor() as cursor:
        sql = "UPDATE teams t SET t.name=%s, t.code=%s, t.active=%s WHERE t.id=%s"
        cursor.execute(sql, (name, code, active, team_id))
    connection.commit()
