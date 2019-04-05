import Services.crawler as crawler
import Persistance.team_repository as team_repo
import Persistance.player_repository as player_repo
import Persistance.season_repository as season_repo
import Persistance.player_season_repository as player_season_repo
from Services.db_connection import connection
from Services.web_drivers import chrome_driver as driver
from Services.timestamp import start_timestamp, end_timestamp
from Data.initial_data import teams, seasons, players, player_season, team_index_by_code


print(start_timestamp())
print('Storing initial data \nSeasons: 2015-16, 2016-17, 2017-18\n')

print('Inserting teams')
team_repo.insert_teams(teams)

print('Inserting players')
player_repo.insert_players(players)

print('Inserting seasons')
season_repo.insert_seasons(seasons)

print('Insert triplets player-team-season')
player_season_repo.insert_multiple_player_season_rows(player_season)

print('Get seasons from DB')
seasons_db = season_repo.get_seasons_by_short_representation(['2016', '2017', '2018'])

print('Insert team games ')
# couple of teams - testing
# crawler.games_for_teams_in_given_seasons(seasons_db, ['CLE', 'LAL'], 21, 23)
# all teams
crawler.games_for_teams_in_given_seasons(seasons_db, list(team_index_by_code.keys()))

print('Insert player games')
players_slugs = [player.slug for player in players]
crawler.games_for_players_in_given_seasons(players_slugs, ['2016', '2017', '2018'])

driver.quit()
connection.close()

print(end_timestamp())
