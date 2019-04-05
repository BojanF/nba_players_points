import json
import pandas as pd
from Persistance.player_repository import get_player_name
from Services.timestamp import start_timestamp, end_timestamp
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif
from Services.csv_service import create_csv_file_for_player, create_csv_file_for_all_players, names
from Feature_selector.selector_functions import rfecv_multiple, percentile_multiple, select_k_best_selector


def create_data_frame(mode, start_date, end_date, player_id=None):
    print('Feature selection data set with games from ', start_date, ' to ', end_date)
    if mode == 1:
        # crete csv file with games from one player
        file_location = create_csv_file_for_player(player_id, start_date, end_date)
        data_frame_res = pd.read_csv(file_location, names=names)
    elif mode == 2:
        # create csv file with games from all players in the system
        file_location = create_csv_file_for_all_players(start_date, end_date)
        data_frame_res = pd.read_csv(file_location, names=names)
    else:
        print('Wrong mode! \n    Enter 1 for matrix creation for given player\n    Enter 2 for matrix creation with'
              ' games from all players')
        return None
    return data_frame_res


def selector_process(x, y, player_id):
    print('Feature selection for', get_player_name(player_id_val))
    # number_of_folds = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    number_of_folds = [50, 55, 60, 65, 70, 75]
    rfevc_res = rfecv_multiple(x, y, number_of_folds)
    print(json.dumps(rfevc_res, indent=4))
    rfevc_res_values = list(rfevc_res.values())
    min_features_with_rfevc = min([val.__len__() for val in rfevc_res_values])
    min_features_with_rfevc = 17

    percentiles_values = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    print('\nPercentile selection with f_classif')
    percentile_multiple(x, y, f_classif, percentiles_values)

    print('\nPercentile selection with f_regression')
    percentile_multiple(x, y, f_regression, percentiles_values)

    print('\nPercentile selection with mutual_info_classif')
    percentile_multiple(x, y, mutual_info_classif, percentiles_values)

    print("\nSelect K best with f_classif")
    f_classif_res = select_k_best_selector(x, y, f_classif, min_features_with_rfevc)
    print('All', f_classif_res[0])
    print('Limited on', min_features_with_rfevc, f_classif_res[1])

    print("\nSelect K best with f_regression")
    f_regression_res = select_k_best_selector(x, y, f_regression, min_features_with_rfevc)
    print('All', f_regression_res[0])
    print('Limited on', min_features_with_rfevc, f_regression_res[1])

    print("\nSelect K best with mutual_info_classif")
    mutual_info_classif_res = select_k_best_selector(x, y, mutual_info_classif, min_features_with_rfevc)
    print('All', mutual_info_classif_res[0])
    print('Limited on', min_features_with_rfevc, mutual_info_classif_res[1])


def selector_process_activation(data, player_id):
    if data is None:
        return
    array = data_frame.values
    x_data = array[:, 0:29]
    y_data = array[:,29]
    selector_process(x_data, y_data, player_id)


print(start_timestamp())

start = '2015-10-27'
end = '2019-04-03'
mode_val = 1
player_id_val = 1
data_frame = create_data_frame(mode_val, start, end, player_id_val)
selector_process_activation(data_frame, player_id_val)

print(end_timestamp())
