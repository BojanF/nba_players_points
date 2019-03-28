import datetime
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif
from Feature_selector.selector_functions import rfecv_selector, percentile_selector, select_k_best_selector
from Feature_selector.csv_file_producer import create_csv_file, names


def percentile_multiple(x, y, function_val, percentiles_array):
    final_result = {}
    for prct in percentiles_array:
        print('\nPercentile selector with', prct, '% features kept')
        print('     Start', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        percentile_result = percentile_selector(x, y, function_val, prct)
        print('     Number of features: ', percentile_result.__len__())
        print('     ', percentile_result)
        final_result[prct] = percentile_result
        print('     End', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return final_result


def rfecv_multiple(x, y, folds):
    final_result = {}
    for number in folds:
        print('\nRFECV with', number, 'folds')
        print('     Start', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        rfecv_result = rfecv_selector(x, y, number)
        print('     Optimal number of features: ', rfecv_result[1])
        print('     RFECV accepted features:')
        print('     ', rfecv_result[0])
        final_result[number] = rfecv_result[0]
        print('     End', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return final_result


print('Start: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

start = '2015-10-27'
end = '2019-03-26'
print('Feature selection data set with games from ', start, ' to ', end)
file_name = create_csv_file(start, end)
data_frame = pd.read_csv('..\\Feature_Selector\\csv_files\\' + file_name, names=names)

array = data_frame.values
x = array[:,0:29]
y = array[:,29]

number_of_folds = [5,10, 15, 20, 25, 30, 35, 40, 45]
rfevc_res = rfecv_multiple(x, y, number_of_folds)
print(rfevc_res)
rfevc_res_values = list(rfevc_res.values())
min_features_with_rfevc = min([val.__len__() for val in rfevc_res_values])
# min_features_with_rfevc = 17

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
print('Limited on', min_features_with_rfevc,  f_regression_res[1])

print("\nSelect K best with mutual_info_classif")
mutual_info_classif_res = select_k_best_selector(x, y, mutual_info_classif, min_features_with_rfevc)
print('All', mutual_info_classif_res[0])
print('Limited on', min_features_with_rfevc,  mutual_info_classif_res[1])

print('\nEnd: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))