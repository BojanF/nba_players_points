# https://www.scikit-yb.org/en/latest/api/target/feature_correlation.html
import pandas as pd
from yellowbrick.target import FeatureCorrelation
from Persistance.player_repository import get_player_name
from Services.timestamp import start_timestamp, end_timestamp
from Services.csv_service import create_csv_file_for_player, create_csv_file_for_all_players, names, fs_names


def feature_correlation(mode, start_date, end_date, feature_selection_mode, player_id=None):

    if mode == 1 and player_id is None:
        print('ERROR\nPlease enter valid player_id')
        return

    print('Feature correlation for player games data set with games from ', start_date, ' to ', end_date)

    number_of_features = names.__len__() - 1
    names_val = names

    if mode == 1 and feature_selection_mode:
        print('Single player, feature selected data')
        # crete csv file with games from one player, feature selected data
        print('Player :', get_player_name(player_id))
        number_of_features = fs_names.__len__() - 1
        names_val = fs_names
        file_location = create_csv_file_for_player(player_id, start_date, end_date, feature_selection_mode)
    elif mode == 1 and (not feature_selection_mode):
        print('Single player')
        # crete csv file with games from one player
        print('Player :', get_player_name(player_id))
        file_location = create_csv_file_for_player(player_id, start_date, end_date)
    elif mode == 2 and feature_selection_mode:
        print('All players, feature selected data')
        # create csv file with games from all players in the system, feature selected data
        number_of_features = fs_names.__len__() - 1
        names_val = fs_names
        file_location = create_csv_file_for_all_players(start_date, end_date, feature_selection_mode)
    elif mode == 2 and (not feature_selection_mode):
        print('All players')
        # create csv file with games from all players in the system
        file_location = create_csv_file_for_all_players(start_date, end_date)
    else:
        print('Wrong mode!\n'
              '    Enter 1 for graph creation for given player\n'
              '    Enter 2 for graph creation with games from all players\n'
              '    True or False value for feature selection mode')
        return

    feature_names = names_val[0:number_of_features]

    data_frame = pd.read_csv(file_location, names=names_val)
    array = data_frame.values
    x = array[:, 0:number_of_features]
    y = array[:, number_of_features]

    # graph 1
    visualizer_1 = FeatureCorrelation(labels=names_val[0:number_of_features])
    visualizer_1.fit(x, y)
    visualizer_1.poof()

    # graph 2
    discrete_features = [False for _ in range(len(names_val[0:number_of_features]))]
    discrete_features[1] = True
    visualizer_2 = FeatureCorrelation(method='mutual_info-regression', labels=names_val[0:number_of_features])
    visualizer_2.fit(x, y, discrete_features=discrete_features, random_state=0)
    visualizer_2.poof()

    # graph 3
    X_pd = pd.DataFrame(x, columns=feature_names)
    visualizer_3 = FeatureCorrelation(method='mutual_info-classification', feature_names=feature_names, sort=True)
    visualizer_3.fit(X_pd, y, random_state=0)
    visualizer_3.poof()


print(start_timestamp() + '\n')

start = '2015-10-27'
end = '2019-04-03'
feature_selection_filtered = False  # True or False value
mode_val = 1  # 1- single player, 2- all players
feature_correlation(mode_val, start, end, feature_selection_filtered, 1)

print(end_timestamp())
