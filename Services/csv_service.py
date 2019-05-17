import os
import csv
from Services.timestamp import start_timestamp_filename_w
import Persistance.player_game_repository as player_game_repo

header = ['id', 'day', 'month', 'year', 'opponent', 'ha_head']
footer = ['pts_result']

features = ['ha', 'minutes_played', 'fg', 'fga', 'fg_pct', 'tp', 'tpa', 'tp_pct', 'ft', 'fta',
         'ft_pct', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts_last_game',
         'game_score_index', 'plus_minus', 'team_win_pct', 'team_streak', 'opponent_win_pct',
         'opponent_streak', 'opponent_streak_in_lg', 'pts_margin', 'under_odd', 'over_odd']
fs_features = ['ha', 'fg_pct', 'tp', 'tp_pct', 'ft', 'fta', 'ft_pct', 'blk', 'opponent_win_pct', 'under_odd', 'over_odd']

names = header + features + footer
fs_names = header + fs_features + footer
header_features = header.__len__()


def create_csv_file_for_all_players(start_date, end_date, fs_mode=None):
    directory = '..\\..\\Files_generated\\CSV_games_files\\All_players\\'
    if fs_mode is not None:
        player_games = player_game_repo.get_player_data_set_games_feature_selected(start_date, end_date)
        directory += 'fs_'
    else:
        player_games = player_game_repo.get_player_games_data_set(start_date, end_date)

    return create_csv_file(directory, player_games)


def create_csv_file_for_player(player_id, start_date, end_date, fs_mode=None):
    directory = '..\\..\\Files_generated\\CSV_games_files\\Player\\'
    if fs_mode is not None:
        player_games = player_game_repo.get_data_set_for_player_feature_selected(player_id, start_date, end_date)
        directory += 'fs_p_id_' + str(player_id) + '_'
    else:
        player_games = player_game_repo.get_data_set_for_player(player_id, start_date, end_date)
        directory += 'p_id_' + str(player_id) + '_'

    return create_csv_file(directory, player_games)


def create_csv_file(directory, games_data):
    print('Number of games: ', games_data.__len__())
    file_name = 'player_games_' + start_timestamp_filename_w() + '.csv'
    file_path = directory + file_name
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', newline='') as out:
        csv_out = csv.writer(out)
        for row in games_data:
            csv_out.writerow(row)
    print('CSV file created\n')
    return file_path


def parse_csv_file(file_path, range_from=1, range_to=82):
    results = []
    with open(file_path) as csvfile:
        data = list(csv.reader(csvfile, quoting=csv.QUOTE_ALL))
        if not (range_from == 1 and range_to == 82):
            data = data[range_from - 1: range_to]
        for row in data:  # each row is a list
            if row[4] == 'x':
                game_info = row[0:4]
                margin_and_odds = [None] * 3
                row = game_info + margin_and_odds
            results.append(row)
    return results

