import datetime
from Data.initial_data import teams, seasons, players, player_season, team_index_by_code
from Services.db_connection import connection
import Services.repository as repo
import Services.crawler as crawler

print('Storing initial data')
print('Start: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Inserting teams')
repo.insert_teams(connection, teams)

print('Inserting players')
repo.insert_players(connection, players)

print('Inserting seasons')
repo.insert_seasons(connection, seasons)

print('Insert pairs player-team for given season')
repo.insert_player_season_pairs(connection, player_season)

print('Get seasons from DB')
seasons_db = repo.get_seasons_by_short_representation(connection, ['2016', '2017', '2018'])

# table from web page selected by xpath
xpathTable = '//*[@id="games"]/tbody'

print('Insert games ')
# couple of teams - testing
# crawler.games_for_team_in_given_seasons(connection, seasons_db, ['CLE', 'LAL'], xpathTable, 1, 2)

# all teams
crawler.games_for_team_in_given_seasons(connection, seasons_db, list(team_index_by_code.keys()), xpathTable)

print('End: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


connection.close()

