import Persistance.player_game_repository as player_game_repo
import datetime
import csv

names = ['ha', 'minutes_played', 'fg', 'fga', 'fg_pct', 'tp', 'tpa', 'tp_pct', 'ft', 'fta',
         'ft_pct', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts_last_game',
         'game_score_index', 'plus_minus', 'team_win_pct', 'team_streak', 'opponent_win_pct',
         'opponent_streak', 'opponent_streak_in_lg', 'pts_margin', 'under_odd', 'over_odd', 'pts_result']


def create_csv_file(start_date, end_date):
    player_games = player_game_repo.get_player_games_for_feature_selection(start_date, end_date)
    print('Number of games: ', player_games.__len__())
    file_name = 'feature_selection ' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.csv'
    with open('..\\Feature_selector\\csv_files\\' + file_name, 'w', newline='') as out:
        csv_out = csv.writer(out)
        for row in player_games:
            csv_out.writerow(row)
    print('CSV file created')
    return file_name

