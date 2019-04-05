from Services.db_connection import connection


def insert(player):
    with connection.cursor() as cursor:
        # Create a new Player record
        sql = "INSERT INTO players (`name`, `lastname`, `birthdate`, `slug`) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (player.name, player.lastname, player.birthdate.strftime('%Y-%m-%d'), player.slug))
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()


def insert_players(players):
    for player in players:
        insert(player)


def get_player_id_by_slug(slug):
    with connection.cursor() as cursor:
        sql = "SELECT p.id FROM players p WHERE p.slug=%s"
        cursor.execute(sql, slug)
        result = cursor.fetchone()
    if result is not None:
        return result[0]
    return None


def get_player_name(player_id):
    with connection.cursor() as cursor:
        sql = """
            select CONCAT(p.name, ' ', p.lastname)
            from players p
            where p.id = %s
        """
        cursor.execute(sql, player_id)
        result = cursor.fetchone()
    if result is not None:
        return result[0]
    return None
