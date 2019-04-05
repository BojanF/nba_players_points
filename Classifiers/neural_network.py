import os
import json
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Persistance.player_repository import get_player_name
from Services.timestamp import start_timestamp, end_timestamp, start_timestamp_filename_w
from Services.csv_service import create_csv_file_for_player, names, fs_names
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


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
            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=20, batch_size=batch_size, verbose=1)
            y_pred = model.predict(X_test)

            # print('y_pred:', y_pred[:5].round())
            # print('y_test:', y_test[:5])

            score = model.evaluate(X_test, y_test, verbose=1)

            # print(score)
            round_score = [round(score[0], 3), round(score[1], 3)]
            sub_results[round_name]['score'] = round_score

            # Confusion matrix
            sub_results[round_name]['conf_matrix'] = confusion_matrix(y_test, y_pred.round()).tolist()

            # Precision
            # print(precision_score(y_test, y_pred.round()))
            sub_results[round_name]['precision_score'] = round(precision_score(y_test, y_pred.round()), 3)

            # Recall
            # print(recall_score(y_test, y_pred.round()))
            sub_results[round_name]['recall_score'] = round(recall_score(y_test, y_pred.round()), 3)

            # F1 score
            # print(f1_score(y_test, y_pred.round()))
            sub_results[round_name]['f1_score'] = round(f1_score(y_test, y_pred.round()), 3)

            # Cohen's kappa
            # print(cohen_kappa_score(y_test, y_pred.round()))
            sub_results[round_name]['cohen_kappa_score'] = round(cohen_kappa_score(y_test, y_pred.round()), 3)

        results[series_name] = sub_results

        round_keys = list(sub_results.keys())
        precision_scores = []
        recall_scores = []
        f1_scores = []
        cohen_kappa_scores = []
        for key in round_keys:
            precision_scores.append(sub_results[key]['precision_score'])
            recall_scores.append(sub_results[key]['recall_score'])
            f1_scores.append(sub_results[key]['f1_score'])
            cohen_kappa_scores.append(sub_results[key]['cohen_kappa_score'])

        results_file.write('     Precisions: ' + json.dumps(precision_scores) + '\n')
        results_file.write('        MIN: ' + str(np.min(precision_scores)) + '\n')
        results_file.write('        MAX: ' + str(np.max(precision_scores)) + '\n')
        results_file.write('        Median: ' + str(np.median(precision_scores)) + '\n')
        results_file.write('        Average:' + str(np.average(precision_scores)) + '\n')

        results_file.write('     Recall scores: ' + json.dumps(recall_scores) + '\n')
        results_file.write('        MIN: ' + str(np.min(recall_scores)) + '\n')
        results_file.write('        MAX: ' + str(np.max(recall_scores)) + '\n')
        results_file.write('        Median: ' + str(np.median(recall_scores)) + '\n')
        results_file.write('        Average:' + str(np.average(recall_scores)) + '\n')

        results_file.write('     F1 scores: ' + json.dumps(f1_scores) + '\n')
        results_file.write('        MIN: ' + str(np.min(f1_scores)) + '\n')
        results_file.write('        MAX: ' + str(np.max(f1_scores)) + '\n')
        results_file.write('        Median: ' + str(np.median(f1_scores)) + '\n')
        results_file.write('        Average:' + str(np.average(f1_scores)) + '\n')

        results_file.write('     Cohen-Kappa: ' + json.dumps(cohen_kappa_scores) + '\n')
        results_file.write('        MIN: ' + str(np.min(cohen_kappa_scores)) + '\n')
        results_file.write('        MAX: ' + str(np.max(cohen_kappa_scores)) + '\n')
        results_file.write('        Median: ' + str(np.median(cohen_kappa_scores)) + '\n')
        results_file.write('        Average:' + str(np.average(cohen_kappa_scores)) + '\n')
        results_file.write('------------------------------------------------------------------------'
                           '------------------------------------------------------------------------\n')

        total_calculation_dict[series_name] = {}
        total_calculation_dict[series_name]['precision_scores'] = precision_scores
        total_calculation_dict[series_name]['recall_scores'] = recall_scores
        total_calculation_dict[series_name]['f1_scores'] = f1_scores
        total_calculation_dict[series_name]['cohen_kappa_scores'] = cohen_kappa_scores

    if rounds[0] > 1:
        print(json.dumps(total_calculation_dict, indent=4))
        complete_results_keys = list(results.keys())
        precision_all_scores = []
        recall_all_scores = []
        f1_all_scores = []
        cohen_kappa_all_scores = []
        for key in complete_results_keys:
            precision_all_scores.append(total_calculation_dict[key]['precision_scores'])
            recall_all_scores.append(total_calculation_dict[key]['recall_scores'])
            f1_all_scores.append(total_calculation_dict[key]['f1_scores'])
            cohen_kappa_all_scores.append(total_calculation_dict[key]['cohen_kappa_scores'])
        results_file.write('Total statistics:\n')
        results_file.write('Precisions: ' + json.dumps(precision_all_scores) + '\n')
        results_file.write('    MIN: ' + str(np.min(precision_all_scores)) + '\n')
        results_file.write('    MAX: ' + str(np.max(precision_all_scores)) + '\n')
        results_file.write('    Median: ' + str(np.median(precision_all_scores)) + '\n')
        results_file.write('    Average:' + str(np.average(precision_all_scores)) + '\n')

        results_file.write('Recall scores: ' + json.dumps(recall_all_scores) + '\n')
        results_file.write('    MIN: ' + str(np.min(recall_all_scores)) + '\n')
        results_file.write('    MAX: ' + str(np.max(recall_all_scores)) + '\n')
        results_file.write('    Median: ' + str(np.median(recall_all_scores)) + '\n')
        results_file.write('    Average:' + str(np.average(recall_all_scores)) + '\n')

        results_file.write('F1 scores: ' + json.dumps(f1_all_scores) + '\n')
        results_file.write('    MIN: ' + str(np.min(f1_all_scores)) + '\n')
        results_file.write('    MAX: ' + str(np.max(f1_all_scores)) + '\n')
        results_file.write('    Median: ' + str(np.median(f1_all_scores)) + '\n')
        results_file.write('    Average:' + str(np.average(f1_all_scores)) + '\n')

        results_file.write('Cohen-Kappa: ' + json.dumps(cohen_kappa_all_scores) + '\n')
        results_file.write('    MIN: ' + str(np.min(cohen_kappa_all_scores)) + '\n')
        results_file.write('    MAX: ' + str(np.max(cohen_kappa_all_scores)) + '\n')
        results_file.write('    Median: ' + str(np.median(cohen_kappa_all_scores)) + '\n')
        results_file.write('    Average:' + str(np.average(cohen_kappa_all_scores)) + '\n')
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

path = '..\\Files_generated\\Classifiers_results\\Neural_networks\\'
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

if feature_selection:
    number_of_features_val = fs_names.__len__() - 1
    file.write('Features used: ' + json.dumps(fs_names) + '\n')
    file.write('Number of features: ' + str(number_of_features_val) + '\n')
    file.write('**Feature selection applied**')
    file_location = create_csv_file_for_player(player_id, start, end, feature_selection)
    data_frame = pd.read_csv(file_location, names=fs_names)
else:
    number_of_features_val = names.__len__() - 1
    file.write('Features used: ' + json.dumps(names) + '\n')
    file.write('Number of features: ' + str(number_of_features_val) + '\n')
    file_location = create_csv_file_for_player(player_id, start, end)
    data_frame = pd.read_csv(file_location, names=names)

number_of_games = data_frame.__len__()
file.write('Number of games: ' + str(number_of_games) + '\n')
batch_size_value = round(number_of_games/10)
file.write('Batch size: ' + str(batch_size_value)+ '\n')

file.close()

array = data_frame.values
X = data_frame.iloc[:, 0:number_of_features_val]
y = array[:, number_of_features_val]

model_results = fit_model(X, y, batch_size_value, test_size_val, repetitions_val, number_of_features_val)

with open(path + result_file_name + '.json', 'w') as outfile:
    json.dump(model_results, outfile, indent=4)

end_timestamp = '\n' + end_timestamp()
file = open(result_file_path, 'a')
file.write(end_timestamp)
file.close()

print(end_timestamp)

