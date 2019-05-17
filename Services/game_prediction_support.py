import numpy as np


# tested
def return_game_set_dates(set_class):
    end_date = '2019-04-10'
    start_date = ''

    if set_class == 1:
        start_date = '2018-10-16'
    elif set_class == 2:
        start_date = '2017-10-17'
    elif set_class == 3:
        start_date = '2016-10-25'
    else:
        start_date = '2015-10-27'

    return start_date, end_date


# tested
def return_threshold_pair(threshold_class):
    if threshold_class == 1:
        high_threshold = 0.9
    elif threshold_class == 2:
        high_threshold = 0.85
    elif threshold_class == 3:
        high_threshold = 0.8
    elif threshold_class == 4:
        high_threshold = 0.75
    else:
        high_threshold = 0.7

    low_threshold = 1.0 - high_threshold
    return high_threshold, round(low_threshold, 2)


# tested
def stats_generator_from_confusion_matrix(confusion_matrix, test_set_size):
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]
    accuracy = round((tp + tn) / test_set_size, 3)
    precision = round(tp / (tp + fp), 3)
    recall = round(tp / (tp + fn), 3)
    f1_score = round((2 * precision * recall) / (precision + recall), 3)

    stats_dict = stats_dictionary_generator(tn, fp, fn, tp, accuracy, precision, recall, f1_score)

    return tn, fp, fn, tp, accuracy, precision, recall, f1_score, stats_dict


# tested
def stats_dictionary_generator(tn, fp, fn, tp, accuracy, precision, recall, f1_score):
    return {
        'conf_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        },
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score)
    }


# tested
def home_away_game(ha):
    if ha == 0:
        return '@'
    return ''


# tested
def conf_matrix_constructor(actual_class, predicted_class):
    matrix = [[0, 0], [0, 0]]
    if actual_class == predicted_class:
        if actual_class == 1:
            matrix[1][1] = 1
        else:
            matrix[0][0] = 1
    else:
        if actual_class == 1 and predicted_class == 0:
            matrix[1][0] = 1
        elif actual_class == 0 and predicted_class == 1:
            matrix[0][1] = 1

    return np.array(matrix)


# tested
def process_single_game_result(prediction_count, voting_intention, game):
    # game[1] is predicted class (1 or 0)
    # game[2] class prediction probability
    # game[3] is bet safe (YES or NO)
    if game[1] == 1:
        voting_intention['voting']['over']['votes'] = voting_intention['voting']['over']['votes'] + 1
        if game[3] == 'YES':
            voting_intention['voting']['over']['bet_safe'] = voting_intention['voting']['over']['bet_safe'] + 1
        voting_intention['voting']['over']['probabilities'][prediction_count] = round(float(game[2]), 3)
    else:
        voting_intention['voting']['under']['votes'] = voting_intention['voting']['under']['votes'] + 1
        if game[3] == 'YES':
            voting_intention['voting']['under']['bet_safe'] = voting_intention['voting']['under']['bet_safe'] + 1
        voting_intention['voting']['under']['probabilities'][prediction_count] = round(float(game[2]), 3)

    return voting_intention


# tested
def process_final_result_for_single_game(actual_class, game):
    final_result = {
        'predicted_class': None,
        'bet_safe': None,
        'conf_matrix': None,
        'bet_safe_conf_matrix': None
    }

    difference = abs(game['voting']['over']['votes'] - game['voting']['under']['votes'])

    if difference > 2 and game['voting']['over']['votes'] > game['voting']['under']['votes']:
        final_result['predicted_class'] = 1
        bet_safe_no_votes = game['voting']['over']['votes'] - game['voting']['over']['bet_safe']
        # bet safe implementation
        if game['voting']['over']['bet_safe'] > bet_safe_no_votes:
            final_result['bet_safe'] = 'YES'
        else:
            final_result['bet_safe'] = 'NO'
    elif difference > 2 and game['voting']['over']['votes'] < game['voting']['under']['votes']:
        final_result['predicted_class'] = 0
        bet_safe_no_votes = game['voting']['under']['votes'] - game['voting']['under']['bet_safe']
        # bet safe implementation
        if game['voting']['under']['bet_safe'] > bet_safe_no_votes:
            final_result['bet_safe'] = 'YES'
        else:
            final_result['bet_safe'] = 'NO'
    elif game['voting']['over']['votes'] > game['voting']['under']['votes']:
        final_result['predicted_class'] = 1
        final_result['bet_safe'] = 'NO'
    else:
        final_result['predicted_class'] = 0
        final_result['bet_safe'] = 'NO'

    final_result['conf_matrix'] = conf_matrix_constructor(actual_class, final_result['predicted_class'])
    final_result['bet_safe_conf_matrix'] = final_result['conf_matrix']
    if final_result['bet_safe'] == 'NO':
        final_result['bet_safe_conf_matrix'] = np.array([[0, 0], [0, 0]])

    return final_result


# print(return_game_set_dates(1))
# print(return_game_set_dates(2))
# print(return_game_set_dates(3))
# print(return_game_set_dates(4))
#
# print(return_threshold_pair(1))
# print(return_threshold_pair(2))
# print(return_threshold_pair(3))
# print(return_threshold_pair(4))
# print(return_threshold_pair(5))
#
# print(conf_matrix_constructor(0, 0))
# print(conf_matrix_constructor(0, 1))
# print(conf_matrix_constructor(1, 0))
# print(conf_matrix_constructor(1, 1))
