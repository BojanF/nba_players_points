# Correction Matrix Plot
#  https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from Feature_selector.csv_file_producer import create_csv_file, names

# renders picture 1.2K x 1.2K pixels
plt.rcParams["figure.figsize"] = (12, 12)

start = '2015-10-27'
end = '2019-03-26'
print('Feature selection data set with games from ', start, ' to ', end)
file_name = create_csv_file(start, end)
data_frame = pd.read_csv('..\\Feature_Selector\\csv_files\\' + file_name, names=names)
correlations = data_frame.corr()

# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,30,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation='vertical')
ax.set_yticklabels(names)
plt.xticks()

# Get current size
fig_size = plt.rcParams["figure.figsize"]

plt.show()

