import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
np.random.seed(1337)  # for reproducibility
from keras import Sequential, backend
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from Persistance.team_repository import get_teams
from Persistance.player_repository import get_player_name
from Services.csv_service import create_csv_file_for_player, names, fs_names, header_features
from Services.timestamp import start_timestamp, end_timestamp, start_timestamp_filename_w, interval_between_dates
from Services.game_prediction_support import home_away_game, stats_generator_from_confusion_matrix, \
    return_game_set_dates, return_threshold_pair

NUM_PARALLEL_EXEC_UNITS = 8
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True,
                        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

session = tf.Session(config=config)

backend.set_session(session)

os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

print('MKL enabled: ', tf.pywrap_tensorflow.IsMklEnabled())


def fit_model(X_train, X_test, y_train, y_test, number_of_features, threshold):
    # building the neural network
    # Initialize the constructor
    model = Sequential()
    # Add an input layer
    model.add(Dense(number_of_features + 1, activation='relu', input_shape=(number_of_features,)))
    # Add one hidden layer
    model.add(Dense(100, activation='relu'))
    # Add an output layer
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
    model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=0, shuffle=False, callbacks=[early_stop])
    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test, verbose=1)
    print(model.predict_classes(X_test, verbose=1))
    predicted_class = model.predict_classes(X_test, verbose=1)[0]

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=[0, 1]).tolist()

    y_pred_val = y_pred[0][predicted_class]  # y_pred = [[0.3, 0.7]]
    print('Y_PRED (probability): ', y_pred_val)
    bet_safe_conf_matrix = conf_matrix
    bet_safe = 'YES'
    if predicted_class == 1 and y_pred_val < threshold:
        bet_safe = 'NO'
        bet_safe_conf_matrix = np.array([[0, 0], [0, 0]])
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

thresholds = return_threshold_pair(threshold_class)
threshold_val = thresholds[0]
set_dates = return_game_set_dates(set_number)
start = set_dates[0]
end = set_dates[1]
player_name = get_player_name(player_id)

model_results = {
    'player': player_name,
    'time_frame': {
        'from': start,
        'to': end
    },
    'model': 'NN 2 softmax - OT',
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

path = '..\\..\\Files_generated\\Classifiers_results\\Game_by_game_threshold\\Neural_network_2\\'
result_file_name = 'p_id_' + str(player_id) + '_' + start_timestamp_filename_w()
result_file_path = path + result_file_name + '.txt'
os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
file = open(result_file_path, 'w')

print('Player:', player_name)
print('Game by game\nNeural network model data set with games from ', start, ' to ', end)

file.write(start_time + '\n')
file.write('\n' + 'Player: ' + player_name + '\n')
file.write('Game by game\nNeural network model softmax\n')
file.write('Data set with games from ' + start + ' to ' + end + '\n')

if feature_selection:
    number_of_features_val = fs_names.__len__() - 1 - header_features
    file.write('Features used: ' + json.dumps(fs_names[header_features:]) + '\n')
    file.write('Number of features: ' + str(number_of_features_val) + '\n')
    file.write('**Feature selection applied**' + '\n')
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
model_results['games_set']['absolute_test_set_size'] = test_set_size
print('Test set size: ', test_set_size)
train_set_size = number_of_games - test_set_size
file.write('Number of games: ' + str(number_of_games) + '\n')
file.write('Test set size: ' + str(test_set_size) + '\n')
file.write('Threshold: ' + str(threshold_val) + '\n')

array = data_frame.values
X_header = data_frame.iloc[:, 0:header_features]
X_data = data_frame.iloc[:, header_features:number_of_features_val+header_features]
y = array[:, number_of_features_val+header_features]
yy = pd.get_dummies(y)
y3 = np.array(yy.values.tolist())
model_results['number_of_features'] = number_of_features_val

games = list(range(0, test_set_size))
master_confusion_matrix = np.array([[0, 0], [0, 0]])
master_bet_safe_confusion_matrix = np.array([[0, 0], [0, 0]])

for idx, g in enumerate(games):
    test_set_end = train_set_size + 1

    X_train_val = X_data[:train_set_size]
    X_test_val = X_data[train_set_size:test_set_end]

    y_train_val = y3[:train_set_size]
    y_test_val_categorical = y3[train_set_size:test_set_end]
    y_test_val_binary = y[train_set_size:test_set_end]

    game_id = X_header.loc[[train_set_size], 'id'][train_set_size]
    day = X_header.loc[[train_set_size], 'day'][train_set_size]
    month = X_header.loc[[train_set_size], 'month'][train_set_size]
    year = X_header.loc[[train_set_size], 'year'][train_set_size]
    ha = X_header.loc[[train_set_size], 'ha_head'][train_set_size]
    game_opponent = X_header.loc[[train_set_size], 'opponent'][train_set_size]

    train_set_size += 1
    print('Game #' + str(idx + 1))

    result = fit_model(X_train_val, X_test_val, y_train_val, y_test_val_categorical, number_of_features_val, threshold_val)
    master_confusion_matrix = master_confusion_matrix + result[0]
    master_bet_safe_confusion_matrix = master_bet_safe_confusion_matrix + result[4]

    game_date = str(day) + '-' + str(month) + '-' + str(year)
    opponent = home_away_game(ha) + teams_dict[game_opponent]
    probability = float(result[2])

    file.write('\n\nGame #' + str(train_set_size) + ' \n')
    file.write('Game ID: ' + str(game_id) + '\n')
    file.write('Date: ' + game_date + '\n')
    file.write('Opponent: ' + opponent + '\n')
    file.write('Actual class:    ' + str(int(y_test_val_binary[0])) + '\n')
    file.write('Predicted class: ' + str(result[1]) + '\n')
    file.write('Probability: ' + str(probability) + '\n')
    file.write('Bet safe: ' + result[3] + '\n')

    # confusion matrix for this game only
    # file.write('\nConfusion matrix: \n')
    # json.dump(result[0][0], file)
    # file.write('\n')
    # json.dump(result[0][1], file)

    final_game_result = {
        'db_id': int(game_id),
        'date': game_date,
        'opponent': opponent,
        'actual_class': int(y_test_val_binary[0]),
        'predicted_class': int(result[1]),
        'probability': probability,
        'bet_safe': result[3]
    }
    model_results['games'][train_set_size] = final_game_result


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
print('Bet safe games: ', number_of_bet_safe_games)
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
