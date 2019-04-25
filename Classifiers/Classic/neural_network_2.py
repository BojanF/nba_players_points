import os
import json
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Persistance.player_repository import get_player_name
from Services.csv_service import create_csv_file_for_player, names, fs_names
from Services.timestamp import start_timestamp, end_timestamp, start_timestamp_filename_w
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def fit_model(x_data, y_data, batch_size, test_size, rounds, number_of_features):
    results_file = open(result_file_path, 'a')
    results = {}
    total_calculation_dict = {}

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=1)

    # Define the scaler
    scaler = StandardScaler().fit(X_train)

    # Scale the train set
    X_train = scaler.transform(X_train)

    # Scale the test set
    X_test = scaler.transform(X_test)

    series = list(range(1, rounds[0]+1))
    iterations = list(range(1, rounds[1]+1))
    for s in series:
        series_name = 'series_' + str(s)
        print(series_name)
        results_file.write(series_name + '\n')
        sub_results = {}
        for itr in iterations:
            round_name = 'round_' + str(itr)
            print('\n', round_name)
            sub_results[round_name] = {}
            # building the neural network
            # Initialize the constructor
            model = Sequential()
            # Add an input layer
            model.add(Dense(number_of_features+1, activation='relu', input_shape=(number_of_features,)))
            # Add one hidden layer
            model.add(Dense(100, activation='relu'))
            # Add an output layer
            model.add(Dense(2, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
            model.fit(X_train, y_train, epochs=20, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[early_stop])
            y_pred = model.predict(X_test)

            score = model.evaluate(X_test, y_test, verbose=1)

            # print(score)
            round_score = [round(score[0], 3), round(score[1], 3)]
            sub_results[round_name]['score'] = round_score

            # Confusion matrix
            sub_results[round_name]['conf_matrix'] = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))\
                .tolist()

            # Accuracy
            sub_results[round_name]['accuracy_score'] = round(accuracy_score(y_test, y_pred.round()), 3)

            # Precision
            sub_results[round_name]['precision_score'] = round(precision_score(y_test, y_pred.round(), average='micro'), 3)

            # Recall
            sub_results[round_name]['recall_score'] = round(recall_score(y_test, y_pred.round(), average='micro'), 3)

            # F1 score
            sub_results[round_name]['f1_score'] = round(f1_score(y_test, y_pred.round(), average='micro'), 3)

        results[series_name] = sub_results

        round_keys = list(sub_results.keys())
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        for key in round_keys:
            accuracy_scores.append(sub_results[key]['accuracy_score'])
            precision_scores.append(sub_results[key]['precision_score'])
            recall_scores.append(sub_results[key]['recall_score'])
            f1_scores.append(sub_results[key]['f1_score'])

        results_file.write('     Accuracy: ' + json.dumps(accuracy_scores) + '\n')
        results_file.write('        MIN: ' + str(np.min(accuracy_scores)) + '\n')
        results_file.write('        MAX: ' + str(np.max(accuracy_scores)) + '\n')
        results_file.write('        Median: ' + str(round(np.median(accuracy_scores), 3)) + '\n')
        results_file.write('        Average: ' + str(round(np.average(accuracy_scores), 3)) + '\n')

        results_file.write('     Precisions: ' + json.dumps(precision_scores) + '\n')
        results_file.write('        MIN: ' + str(np.min(precision_scores)) + '\n')
        results_file.write('        MAX: ' + str(np.max(precision_scores)) + '\n')
        results_file.write('        Median: ' + str(round(np.median(precision_scores), 3)) + '\n')
        results_file.write('        Average: ' + str(round(np.average(precision_scores), 3)) + '\n')

        results_file.write('     Recall scores: ' + json.dumps(recall_scores) + '\n')
        results_file.write('        MIN: ' + str(np.min(recall_scores)) + '\n')
        results_file.write('        MAX: ' + str(np.max(recall_scores)) + '\n')
        results_file.write('        Median: ' + str(round(np.median(recall_scores), 3)) + '\n')
        results_file.write('        Average: ' + str(round(np.average(recall_scores), 3)) + '\n')

        results_file.write('     F1 scores: ' + json.dumps(f1_scores) + '\n')
        results_file.write('        MIN: ' + str(np.min(f1_scores)) + '\n')
        results_file.write('        MAX: ' + str(np.max(f1_scores)) + '\n')
        results_file.write('        Median: ' + str(round(np.median(f1_scores), 3)) + '\n')
        results_file.write('        Average: ' + str(round(np.average(f1_scores), 3)) + '\n')

        results_file.write('------------------------------------------------------------------------'
                           '------------------------------------------------------------------------\n')

        total_calculation_dict[series_name] = {}
        total_calculation_dict[series_name]['accuracy_scores'] = accuracy_scores
        total_calculation_dict[series_name]['precision_scores'] = precision_scores
        total_calculation_dict[series_name]['recall_scores'] = recall_scores
        total_calculation_dict[series_name]['f1_scores'] = f1_scores

    if rounds[0] > 1:
        print(json.dumps(total_calculation_dict, indent=4))
        complete_results_keys = list(results.keys())
        accuracy_all_scores = []
        precision_all_scores = []
        recall_all_scores = []
        f1_all_scores = []
        for key in complete_results_keys:
            accuracy_all_scores.append(total_calculation_dict[key]['accuracy_scores'])
            precision_all_scores.append(total_calculation_dict[key]['precision_scores'])
            recall_all_scores.append(total_calculation_dict[key]['recall_scores'])
            f1_all_scores.append(total_calculation_dict[key]['f1_scores'])

        results_file.write('Total statistics:\n')
        results_file.write('Accuracy: ' + json.dumps(accuracy_all_scores) + '\n')
        results_file.write('    MIN: ' + str(np.min(accuracy_all_scores)) + '\n')
        results_file.write('    MAX: ' + str(np.max(accuracy_all_scores)) + '\n')
        results_file.write('    Median: ' + str(round(np.median(accuracy_all_scores), 3)) + '\n')
        results_file.write('    Average: ' + str(round(np.average(accuracy_all_scores), 3)) + '\n')

        results_file.write('Precisions: ' + json.dumps(precision_all_scores) + '\n')
        results_file.write('    MIN: ' + str(np.min(precision_all_scores)) + '\n')
        results_file.write('    MAX: ' + str(np.max(precision_all_scores)) + '\n')
        results_file.write('    Median: ' + str(round(np.median(precision_all_scores), 3)) + '\n')
        results_file.write('    Average: ' + str(round(np.average(precision_all_scores), 3)) + '\n')

        results_file.write('Recall scores: ' + json.dumps(recall_all_scores) + '\n')
        results_file.write('    MIN: ' + str(np.min(recall_all_scores)) + '\n')
        results_file.write('    MAX: ' + str(np.max(recall_all_scores)) + '\n')
        results_file.write('    Median: ' + str(round(np.median(recall_all_scores), 3)) + '\n')
        results_file.write('    Average: ' + str(round(np.average(recall_all_scores), 3)) + '\n')

        results_file.write('F1 scores: ' + json.dumps(f1_all_scores) + '\n')
        results_file.write('    MIN: ' + str(np.min(f1_all_scores)) + '\n')
        results_file.write('    MAX: ' + str(np.max(f1_all_scores)) + '\n')
        results_file.write('    Median: ' + str(round(np.median(f1_all_scores), 3)) + '\n')
        results_file.write('    Average: ' + str(round(np.average(f1_all_scores), 3)) + '\n')

    results_file.close()
    return results


# START
start_time = start_timestamp()
print(start_time)

feature_selection = False  # True or False value
start = '2015-10-27'
end = '2019-04-03'
test_size_val = 0.30
player_id = 1
repetitions_val = (5, 20)

path = '..\\..\\Files_generated\\Classifiers_results\\Classic\\Neural_network_2\\'
result_file_name = 'p_id_' + str(player_id) + '_' + start_timestamp_filename_w()
result_file_path = path + result_file_name + '.txt'
os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
file = open(result_file_path, 'w')

print('Player:', get_player_name(player_id))
print('Number of series/rounds:', repetitions_val)
print('Neural network model data set with games from ', start, ' to ', end)

file.write(start_time + '\n')
file.write('\n' + 'Player: ' + get_player_name(player_id) + '\n')
file.write('Number of series: ' + str(repetitions_val[0]) + '\n')
file.write('Number of rounds in series: ' + str(repetitions_val[1]) + '\n')
file.write('Neural network model data set with games from ' + start + ' to ' + end + '\n')


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
file.write('Number of games: ' + str(number_of_games) + '\n')
batch_size_value = round(number_of_games/10)
file.write('Batch size: ' + str(batch_size_value)+ '\n')

file.close()

array = data_frame.values
X_header = data_frame.iloc[:, 0:header_features]
X_data = data_frame.iloc[:, header_features:number_of_features_val+header_features]
y = array[:, number_of_features_val+header_features]
yy = pd.get_dummies(y)
y3 = np.array(yy.values.tolist())

model_results = fit_model(X_data, y3, batch_size_value, test_size_val, repetitions_val, number_of_features_val)

with open(path + result_file_name + '.json', 'w') as outfile:
    json.dump(model_results, outfile, indent=4)

end_timestamp = '\n' + end_timestamp()
file = open(result_file_path, 'a')
file.write(end_timestamp)
file.close()

print(end_timestamp)

