from yellowbrick.target import FeatureCorrelation
from Feature_selector.csv_file_producer import create_csv_file, names
import pandas as pd
import datetime

print('Start: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

feature_names = names[0:29]

start = '2015-10-27'
end = '2019-03-26'
print('Feature selection data set with games from ', start, ' to ', end)
file_name = create_csv_file(start, end)
data_frame = pd.read_csv('..\\Feature_Selector\\csv_files\\' + file_name, names=names)
array = data_frame.values
x = array[:,0:29]
y = array[:,29]

# graph 1
visualizer_1 = FeatureCorrelation(labels=names[0:29])
visualizer_1.fit(x, y)
visualizer_1.poof()

# graph 2
discrete_features = [False for _ in range(len(names[0:29]))]
discrete_features[1] = True
visualizer_2 = FeatureCorrelation(method='mutual_info-regression', labels=names[0:29])
visualizer_2.fit(x, y, discrete_features=discrete_features, random_state=0)
visualizer_2.poof()


# graph 3
X_pd = pd.DataFrame(x, columns=feature_names)
visualizer_3 = FeatureCorrelation(method='mutual_info-classification', feature_names=feature_names, sort=True)
visualizer_3.fit(X_pd, y, random_state=0)
visualizer_3.poof()
print('\nEnd: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))