from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, SelectKBest, SelectPercentile, RFECV
from sklearn.svm import SVR
from Feature_selector.csv_file_producer import names


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

