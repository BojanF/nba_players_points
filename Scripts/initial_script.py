import datetime
from Data.initial_data import teams, seasons, players, player_season, team_index_by_code
from Services.db_connection import connection
import Services.repository as repo
import Services.crawler as crawler

print('Storing initial data')
print('Start: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Inserting teams')
repo.insert_teams(teams)

print('Inserting players')
repo.insert_players(players)

print('Inserting seasons')
repo.insert_seasons(seasons)

print('Insert pairs player-team for given season')
repo.insert_player_season_pairs(player_season)

print('Get seasons from DB')
seasons_db = repo.get_seasons_by_short_representation(['2016', '2017', '2018'])

# table from web page selected by xpath
team_xpath_table = '//*[@id="games"]/tbody'

print('Insert team games ')
# couple of teams - testing
# crawler.games_for_team_in_given_seasons(seasons_db, ['CLE', 'LAL'], team_xpath_table, 1, 5)

# all teams
crawler.games_for_team_in_given_seasons(seasons_db, list(team_index_by_code.keys()), team_xpath_table)

print('Insert player games')
player_xpath_table = '//*[@id="pgl_basic"]/tbody'
players_slugs = [player.slug for player in players]
crawler.games_for_players_in_given_seasons(players_slugs, [2016, 2017, 2018], player_xpath_table, 1, 10)

connection.close()

print('End: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
