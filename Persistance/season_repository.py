from Services.db_connection import connection


def insert(season):
    with connection.cursor() as cursor:
        # Create a new Season record
        sql = "INSERT INTO seasons (`name`, short) VALUES (%s, %s)"
        cursor.execute(sql, (season.full, season.short))
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()


def insert_seasons(seasons):
    for season in seasons:
        insert(season)


def get_season_by_short_representation(season_short):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM seasons s WHERE s.short=%s"
        cursor.execute(sql, season_short)
        result = cursor.fetchone()
    return result


def get_seasons_by_short_representation(seasons_short):
    result = [get_season_by_short_representation(season) for season in seasons_short]
    return result
