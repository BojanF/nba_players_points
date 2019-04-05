#  https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Persistance.player_repository import get_player_name
from Services.csv_service import create_csv_file_for_player, create_csv_file_for_all_players, names, fs_names


def correlation_matrix_matplotlib(mode, start_date, end_date, feature_selection_mode, player_id=None):

    if mode == 1 and player_id is None:
        print('ERROR\nPlease enter valid player_id')
        return

    number_of_features = names.__len__()
    names_val = names

    # renders picture 1.2K x 1.2K pixels
    plt.rcParams["figure.figsize"] = (12, 12)

    print('Player games data set with games from ', start_date, ' to ', end_date)

    if mode == 1 and feature_selection_mode:
        print('Single player, feature selected data')
        # crete csv file with games from one player, feature selected data
        number_of_features = fs_names.__len__()
        names_val = fs_names
        file_location = create_csv_file_for_player(player_id, start_date, end_date, feature_selection_mode)
        title = get_player_name(player_id) + ' - FS'
    elif mode == 1 and (not feature_selection_mode):
        print('Single player')
        # crete csv file with games from one player
        file_location = create_csv_file_for_player(player_id, start_date, end_date)
        title = get_player_name(player_id)
    elif mode == 2 and feature_selection_mode:
        print('All players, feature selected data')
        # create csv file with games from all players in the system, feature selected data
        number_of_features = fs_names.__len__()
        names_val = fs_names
        file_location = create_csv_file_for_all_players(start_date, end_date, feature_selection_mode)
        title = 'All players - FS'
    elif mode == 2 and (not feature_selection_mode):
        print('All players')
        # create csv file with games from all players in the system
        file_location = create_csv_file_for_all_players(start_date, end_date)
        title = 'All players'
    else:
        print('Wrong mode!\n'
              '     Enter mode 1 for matrix creation for given player\n'
              '     Enter mode 2 for matrix creation with games from all players\n'
              '     True or False value for feature selection mode')
        return
    data_frame = pd.read_csv(file_location, names=names_val)
    correlations = data_frame.corr()

    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    fig.suptitle(title, fontsize=26)
    ticks = np.arange(0, number_of_features, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names_val, rotation='vertical')
    ax.set_yticklabels(names_val)
    plt.xticks()
    # show plot
    plt.show()


start = '2015-10-27'
end = '2019-04-03'
feature_selection = False   # True or False value
mode_val = 1  # 1- single player, 2- all players
correlation_matrix_matplotlib(mode_val, start, end, feature_selection, 2)

