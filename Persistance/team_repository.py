from Services.db_connection import connection


def insert(team):
    with connection.cursor() as cursor:
        # Create a new Team record
        sql = "INSERT INTO teams (`name`, `code`) VALUES (%s, %s)"
        cursor.execute(sql, (team.name, team.code))
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


def get_teams():
    with connection.cursor() as cursor:
        sql = "SELECT * FROM teams"
        cursor.execute(sql)
        result = cursor.fetchall()
    return result

