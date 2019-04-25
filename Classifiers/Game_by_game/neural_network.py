import os
import json
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from Persistance.team_repository import get_teams
from Persistance.player_repository import get_player_name
from Services.csv_service import create_csv_file_for_player, names, fs_names
from Services.timestamp import start_timestamp, end_timestamp, start_timestamp_filename_w


def home_away_game(ha):
    if ha == 0:
        return '@'
    return ''


def fit_model(X_train, X_test, y_train, y_test, number_of_features):
    # building the neural network
    # Initialize the constructor
    model = Sequential()
    # Add an input layer
    model.add(Dense(number_of_features + 1, activation='relu', input_shape=(number_of_features,)))
    # Add one hidden layer
    model.add(Dense(100, activation='relu'))
    # Add an output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
    model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test, verbose=1)

    predicted_class = model.predict_classes(X_test, verbose=1)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred.round(), labels=[0, 1]).tolist()

    return conf_matrix, predicted_class[0][0]


# START
start_time = start_timestamp()
print(start_time)

feature_selection = False  # True or False value
start = '2015-10-27'
end = '2019-04-03'
test_set_size = 0.30
player_id = 1

teams_dict = {}
teams = get_teams()
for t in teams:
    teams_dict[t[0]] = t[1]

path = '..\\..\\Files_generated\\Classifiers_results\\Game_by_game\\Neural_network\\'
result_file_name = 'p_id_' + str(player_id) + '_' + start_timestamp_filename_w()
result_file_path = path + result_file_name + '.txt'
os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
file = open(result_file_path, 'w')

print('Player:', get_player_name(player_id))
print('Game by game neural network model data set with games from ', start, ' to ', end)

file.write(start_time + '\n')
file.write('\n' + 'Player: ' + get_player_name(player_id) + '\n')
file.write('Game by game\nNeural network model\n')
file.write('Data set with games from ' + start + ' to ' + end + '\n')

header_features = 6

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
test_set_size = round(number_of_games * test_set_size)
train_set_size = number_of_games - test_set_size
file.write('Number of games: ' + str(number_of_games) + '\n')
file.write('Test set size: ' + str(test_set_size) + '\n')

array = data_frame.values
X_header = data_frame.iloc[:, 0:header_features]
X_data = data_frame.iloc[:, header_features:number_of_features_val+header_features]
y = array[:, number_of_features_val+header_features]

games = list(range(0, test_set_size))
master_confusion_matrix = np.array([[0, 0], [0, 0]])

for g in games:
    test_set_end = train_set_size + 1

    X_train_val = X_data[:train_set_size]
    X_test_val = X_data[train_set_size:test_set_end]

    y_train_val = y[:train_set_size]
    y_test_val = y[train_set_size:test_set_end]

    # print(X_header.loc[[train_set_size], ['id', 'date', 'opponent']])

    game_id = X_header.loc[[train_set_size], 'id'][train_set_size]
    day = X_header.loc[[train_set_size], 'day'][train_set_size]
    month = X_header.loc[[train_set_size], 'month'][train_set_size]
    year = X_header.loc[[train_set_size], 'year'][train_set_size]
    ha = X_header.loc[[train_set_size], 'ha_head'][train_set_size]
    game_opponent = X_header.loc[[train_set_size], 'opponent'][train_set_size]

    train_set_size += 1

    result = fit_model(X_train_val, X_test_val, y_train_val, y_test_val, number_of_features_val)
    master_confusion_matrix = master_confusion_matrix + result[0]
    file.write('\n\nGame #' + str(train_set_size) + ' \n')
    file.write('Game ID: ' + str(game_id) + '\n')
    file.write('Date: ' + str(day) + '/' + str(month) + '/' + str(year) + '\n')
    file.write('Opponent: ' + home_away_game(ha) + teams_dict[game_opponent] + '\n')
    file.write('Actual class:    ' + str(int(y_test_val[0])) + '\n')
    file.write('Predicted class: ' + str(result[1]))

    # confusion matrix for this game only
    # file.write('\nConfusion matrix: \n')
    # json.dump(result[0][0], file)
    # file.write('\n')
    # json.dump(result[0][1], file)


tn = master_confusion_matrix[0][0]
fp = master_confusion_matrix[0][1]
fn = master_confusion_matrix[1][0]
tp = master_confusion_matrix[1][1]
accuracy = round((tp + tn ) / test_set_size, 3)
precision = round(tp / (tp + fp), 3)
recall = round(tp / (tp + fn), 3)
f1_score = round((2*precision*recall) / (precision + recall), 3)


print('\nConfusion matrix: ')
print(master_confusion_matrix)
print('Accuracy: ', accuracy)
print('Precision:', precision)
print('Recall: ', recall)
print('F1 score: ', f1_score)


file.write('\n\nConfusion matrix: \n')
json.dump(master_confusion_matrix[0].tolist(), file)
file.write('\n')
json.dump(master_confusion_matrix[1].tolist(), file)
file.write('\nAccuracy: ' + str( accuracy) + '\n')
file.write('Precision: ' + str(precision) + '\n')
file.write('Recall: ' + str( recall) + '\n')
file.write('F1 score: ' + str( f1_score) + '\n')

end_timestamp = '\n' + end_timestamp()
file.write(end_timestamp)
file.close()

print(end_timestamp)
