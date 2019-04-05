import Services.crawler as crawler
import Persistance.team_repository as team_repo
import Persistance.game_repository as game_repo
import Persistance.season_repository as season_repo
import Persistance.player_repository as player_repo
import Persistance.player_game_repository as player_game_repo
from Services.db_connection import connection
from Services.web_drivers import chrome_driver as driver
from Services.timestamp import start_timestamp, end_timestamp


def games_in_given_season_for_all_teams(season_id):
    teams = team_repo.get_active_teams()
    count = 0
    for team in teams:
        games = game_repo.count_games_for_team_in_given_season(team[0], season_id)
        count += games
        print(str(games) + ' - ' + team[2] + ' : ' + team[1])
    print(str(count) + '\n')


def add_additional_games_for_teams(season):
    print(start_timestamp())

    season = season_repo.get_season_by_short_representation(season)
    teams = team_repo.get_active_teams()

    for team in teams:
        games = game_repo.count_games_for_team_in_given_season(team[0], season[0])
        crawler.games_for_team(season, team[2], games + 1)
    print(end_timestamp() + '\n')


def add_additional_games_for_players(season, players):
    print(start_timestamp() + '\n')

    season_db = season_repo.get_season_by_short_representation(season)
    for player in players:
        player_id = player_repo.get_player_id_by_slug(player)
        if player_id is None:
            print('Player with slug ' + player + ' does not exist')
            continue
        games = player_game_repo.count_games_for_player_in_season(player_id, season_db[0])
        print('     Before: ' + player + ' ' + str(games) + ' games')
        crawler.games_for_player(player, season, games+1)
        games_after = player_game_repo.count_games_for_player_in_season(player_id, season_db[0])
        print('     After: ' + player + ' ' + str(games_after) + ' games\n')
    print(end_timestamp())


# add new games for teams in given season
season_id_val = 4
print('Games per team before addition:')
games_in_given_season_for_all_teams(season_id_val)
add_additional_games_for_teams('2019')
print('Games per team after addition:')
games_in_given_season_for_all_teams(season_id_val)

# add new games for player in given season
# prior to this
# you have to add pts margins and odds for newly played games in csv files
# games that you will add
# and
# you have to run add_additional_games_for_teams() function
players_slugs = ['jamesle01', 'derozde01', 'westbru01', 'jokicni01', 'doncilu01', 'kuzmaky01']
print('Addition games for players:')
add_additional_games_for_players('2019', players_slugs)

driver.quit()
connection.close()
