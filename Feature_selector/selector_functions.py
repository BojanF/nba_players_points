from sklearn.svm import SVR
from Services.csv_service import names
from Services.timestamp import start_timestamp, end_timestamp
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFECV


def rfecv_selector(X, Y, cv_value=10):
    estimator = SVR(kernel='linear')
    selector = RFECV(estimator=estimator, cv=cv_value)
    selector.fit(X, Y)
    number_of_features = selector.n_features_
    features_selected = []
    for index, val in enumerate(selector.support_):
        if val:
            features_selected.append(names[index])
    return features_selected, number_of_features


def select_k_best_selector(X, Y, score_func_val, k_val):
    selector = SelectKBest(score_func=score_func_val, k=k_val)
    selector_res = selector.fit(X, Y)
    scores_names = {}
    scores = selector_res.scores_
    for index, val in enumerate(scores):
        scores_names[names[index]] = round(val, 3)
    sorted_results = sorted(scores_names.items(), key=lambda kv: kv[1])
    return sorted_results, sorted_results[-k_val:]


def percentile_selector(X, Y, function_val, percentile_value=10):
    Selector_f = SelectPercentile(function_val, percentile=percentile_value)
    X_new = Selector_f.fit_transform(X,Y)
    dictionary = {}
    for n,s in zip(names,Selector_f.scores_):
        dictionary[n] = round(s, 3)
    sorted_percentiles = sorted(dictionary.items(), key=lambda kv: kv[1])
    return sorted_percentiles[-X_new.shape[1]:]


def percentile_multiple(x, y, function_val, percentiles_array):
    final_result = {}
    for prct in percentiles_array:
        print('\nPercentile selector with', prct, '% features kept')
        print('     ', start_timestamp())
        percentile_result = percentile_selector(x, y, function_val, prct)
        print('     Number of features: ', percentile_result.__len__())
        print('     ', percentile_result)
        final_result[prct] = percentile_result
        print('     ', end_timestamp())
    return final_result


def rfecv_multiple(x, y, folds):
    final_result = {}
    for number in folds:
        print('\nRFECV with', number, 'folds')
        print('     ', start_timestamp())
        rfecv_result = rfecv_selector(x, y, number)
        print('     Optimal number of features: ', rfecv_result[1])
        print('     RFECV accepted features:', rfecv_result[0])
        final_result[number] = rfecv_result[0]
        print('     ', end_timestamp())
    return final_result
