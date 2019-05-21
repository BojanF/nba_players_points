import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from Persistance.team_repository import get_teams
from sklearn.ensemble import RandomForestClassifier
from Persistance.player_repository import get_player_name
from Services.csv_service import create_csv_file_for_player, names, fs_names, header_features
from Services.timestamp import start_timestamp, end_timestamp, start_timestamp_filename_w, interval_between_dates
from Services.game_prediction_support import process_final_result_for_single_game, process_single_game_result,\
    home_away_game, stats_generator_from_confusion_matrix, return_game_set_dates, return_threshold_pair


def fit_model(X_train, X_test, y_train, y_test, threshold):
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=300)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    classifier = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # print('Y_PRED: ', res1.predict_proba(X_test))
    predicted_class = int(clf.predict(X_test)[0])

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred.round(), labels=[0, 1]).tolist()

    # Probabilities
    classes_probabilities = classifier.predict_proba(X_test)
    print('PRB: ', classes_probabilities)

    y_pred_val = classes_probabilities[0][predicted_class]
    print('Y_PRED: ', y_pred_val)
    bet_safe_conf_matrix = conf_matrix
    bet_safe = 'YES'
    if predicted_class == 1 and y_pred_val < threshold:
        bet_safe = 'NO'
        bet_safe_conf_matrix = np.array([[0, 0], [0, 0]])
    # elif predicted_class == 0 and y_pred_val > low_threshold:
    elif predicted_class == 0 and y_pred_val < threshold:
        bet_safe = 'NO'
        bet_safe_conf_matrix = np.array([[0, 0], [0, 0]])

    return conf_matrix, predicted_class, y_pred_val, bet_safe, bet_safe_conf_matrix


# START
start_time = start_timestamp()
print(start_time)

feature_selection = False  # True or False value
set_number = 1
threshold_class = 1
player_id = 1
test_set_size = 0.30
predictions_per_game = 7  # minimum 5 and has to be odd number

set_dates = return_game_set_dates(set_number)
start = set_dates[0]
end = set_dates[1]
player_name = get_player_name(player_id)
thresholds = return_threshold_pair(threshold_class)
threshold_val = thresholds[0]

model_results = {
    'player': player_name,
    'time_frame': {
        'from': start,
        'to': end
    },
    'model': 'random forest - voting',
    'duration': None,
    'feature_selection': feature_selection,
    'number_of_features': 0,
    'threshold': threshold_val,
    'games_set': {
        'games': 0,
        'absolute_test_set_size': 0,
        'relative_test_set_size': test_set_size
    },
    'games': {},
    'results': {
        'general': {},
        'bet_safe': {}
    }
}

teams_dict = {}
teams = get_teams()
for t in teams:
    teams_dict[t[0]] = t[1]

path = '..\\..\\Files_generated\\Classifiers_results\\Game_by_game_threshold_voting\\Random_forest\\'
result_file_name = 'p_id_' + str(player_id) + '_' + start_timestamp_filename_w()
result_file_path = path + result_file_name + '.txt'
os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
file = open(result_file_path, 'w')

print('Player:', player_name)
print('Game by game\nRandom forest model')
print('Data set with games from ', start, ' to ', end)

file.write(start_time + '\n')
file.write('\n' + 'Player: ' + player_name + '\n')
file.write('Game by game\nRandom forest model\n')
file.write('Data set with games from ' + start + ' to ' + end + '\n')

if feature_selection:
    number_of_features_val = fs_names.__len__() - 1 - header_features
    file.write('Features used: ' + json.dumps(fs_names[header_features:]) + '\n')
    file.write('Number of features: ' + str(number_of_features_val) + '\n')
    file.write('**Feature selection applied**')
    file_location = create_csv_file_for_player(player_id, start, end, feature_selection)
    data_frame = pd.read_csv(file_location, names=fs_names)
else:
    number_of_features_val = names.__len__() - 1 - header_features
    file.write('Features used: ' + json.dumps(names[header_features:]) + '\n')
    file.write('Number of features: ' + str(number_of_features_val) + '\n')
    file_location = create_csv_file_for_player(player_id, start, end)
    data_frame = pd.read_csv(file_location, names=names)

number_of_games = data_frame.__len__()
model_results['games_set']['games'] = number_of_games
test_set_size = round(number_of_games * test_set_size)
print('Test set size: ', test_set_size)
model_results['games_set']['absolute_test_set_size'] = test_set_size
train_set_size = number_of_games - test_set_size
file.write('Number of games: ' + str(number_of_games) + '\n')
file.write('Test set size: ' + str(test_set_size) + '\n')
file.write('Threshold: ' + str(threshold_val) + '\n')

array = data_frame.values
X_header = data_frame.iloc[:, 0:header_features]
X_data = data_frame.iloc[:, header_features:number_of_features_val+header_features]
y = array[:, number_of_features_val+header_features]
model_results['number_of_features'] = number_of_features_val

games = list(range(0, test_set_size))
master_confusion_matrix = np.array([[0, 0], [0, 0]])
master_bet_safe_confusion_matrix = np.array([[0, 0], [0, 0]])

for idx, g in enumerate(games):
    st = start_timestamp()
    print(st)
    test_set_end = train_set_size + 1

    X_train_val = X_data[:train_set_size]
    X_test_val = X_data[train_set_size:test_set_end]

    y_train_val = y[:train_set_size]
    y_test_val = y[train_set_size:test_set_end]

    game_id = int(X_header.loc[[train_set_size], 'id'][train_set_size])  # int for json serialization
    day = X_header.loc[[train_set_size], 'day'][train_set_size]
    month = X_header.loc[[train_set_size], 'month'][train_set_size]
    year = X_header.loc[[train_set_size], 'year'][train_set_size]
    ha = X_header.loc[[train_set_size], 'ha_head'][train_set_size]
    game_opponent = X_header.loc[[train_set_size], 'opponent'][train_set_size]

    train_set_size += 1
    single_game_votes = {
        'voting': {
            'over': {
                'votes': 0,
                'bet_safe': 0,
                'probabilities': {}
            },
            'under': {
                'votes': 0,
                'bet_safe': 0,
                'probabilities': {}
            }
        },
        'game': {
            'db_id': 1,
            'date': None,
            'opponent': None,
            'actual_class': None,
            'predicted_class': None,
            'bet_safe': None
        }
    }

    for i in range(1, predictions_per_game+1):
        single_game_result = fit_model(X_train_val, X_test_val, y_train_val, y_test_val, threshold_val)
        single_game_votes = process_single_game_result(i, single_game_votes, single_game_result)

    single_game_overall_res = process_final_result_for_single_game(int(y_test_val[0]), single_game_votes)

    game_in_test_set = str(train_set_size)
    game_date = str(day) + '-' + str(month) + '-' + str(year)
    opponent = home_away_game(ha) + teams_dict[game_opponent]
    final_predicted_class = single_game_overall_res['predicted_class']
    final_bet_safe = single_game_overall_res['bet_safe']

    # confusion matrices
    master_confusion_matrix = master_confusion_matrix + single_game_overall_res['conf_matrix']
    master_bet_safe_confusion_matrix = master_bet_safe_confusion_matrix + single_game_overall_res['bet_safe_conf_matrix']

    file.write('\n\nGame #' + game_in_test_set + ' \n')
    file.write(st + '\n')
    file.write('Game ID: ' + str(game_id) + '\n')
    print('Game ID: ' + str(game_id))
    print('Game #' + str(idx+1))
    file.write('Date: ' + game_date + '\n')
    file.write('Opponent: ' + opponent + '\n')
    file.write('Actual class:    ' + str(int(y_test_val[0])) + '\n')
    file.write('Predicted class: ' + str(final_predicted_class) + '\n')
    file.write('Bet safe: ' + final_bet_safe + '\n')
    en = end_timestamp()
    file.write(en + '\n-----------------------------------------')
    print(en + '\n-----------------------------------------------------------------------------')

    final_game_result = {
        'db_id': game_id,
        'date': game_date,
        'opponent': opponent,
        'actual_class': int(y_test_val[0]),
        'predicted_class': final_predicted_class,
        'bet_safe': final_bet_safe
    }
    single_game_votes['game'] = final_game_result
    model_results['games'][game_in_test_set] = single_game_votes

    # confusion matrix for this game only
    # file.write('\nConfusion matrix: \n')
    # json.dump(result[0][0], file)
    # file.write('\n')
    # json.dump(result[0][1], file)


# general stats
tn, fp, fn, tp, accuracy, precision, recall, f1_score, general_stats_dict = \
    stats_generator_from_confusion_matrix(master_confusion_matrix, test_set_size)
model_results['results']['general'] = general_stats_dict

print('\nConfusion matrix: ')
print(master_confusion_matrix)
print('Accuracy: ', accuracy)
print('Precision:', precision)
print('Recall: ', recall)
print('F1 score: ', f1_score)

file.write('\n\nGeneral stats: \n')
file.write('Confusion matrix: \n')
json.dump(master_confusion_matrix[0].tolist(), file)
file.write('\n')
json.dump(master_confusion_matrix[1].tolist(), file)
file.write('\nAccuracy: ' + str(accuracy) + '\n')
file.write('Precision: ' + str(precision) + '\n')
file.write('Recall: ' + str(recall) + '\n')
file.write('F1 score: ' + str(f1_score) + '\n')


# bet safe stats
number_of_bet_safe_games = np.sum(master_bet_safe_confusion_matrix)
bs_tn, bs_fp, bs_fn, bs_tp, bs_accuracy, bs_precision, bs_recall, bs_f1_score, bet_safe_stats_dict = \
    stats_generator_from_confusion_matrix(master_bet_safe_confusion_matrix, number_of_bet_safe_games)
bet_safe_stats_dict['number_of_games'] = int(number_of_bet_safe_games)
bet_safe_stats_dict['number_of_games_pct'] = str(round((number_of_bet_safe_games / test_set_size) * 100, 2)) + '%'
model_results['results']['bet_safe'] = bet_safe_stats_dict

print('\nBet safe stats')
print('Bet safe games: ', str(number_of_bet_safe_games))
print('Confusion matrix: ')
print(master_bet_safe_confusion_matrix)
print('Accuracy: ', bs_accuracy)
print('Precision:', bs_precision)
print('Recall: ', bs_recall)
print('F1 score: ', bs_f1_score)

file.write('\n\nBet safe stats: \n')
file.write('Bet safe games: ' + str(number_of_bet_safe_games) + '\n')
file.write('Confusion matrix: \n')
json.dump(master_bet_safe_confusion_matrix[0].tolist(), file)
file.write('\n')
json.dump(master_bet_safe_confusion_matrix[1].tolist(), file)
file.write('\nAccuracy: ' + str(bs_accuracy) + '\n')
file.write('Precision: ' + str(bs_precision) + '\n')
file.write('Recall: ' + str(bs_recall) + '\n')
file.write('F1 score: ' + str(bs_f1_score) + '\n')


end_time = '\n' + end_timestamp()
duration = interval_between_dates(start_time, end_time)
model_results['duration'] = duration

with open(path + result_file_name + '.json', 'w') as outfile:
    json.dump(model_results, outfile, indent=4)

file.write(end_time)
file.close()

print(end_time)
print('\nDuration: ', duration)
