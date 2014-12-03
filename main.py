import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import svm, grid_search
#from boostedDT import BoostedDT
from datetime import datetime
from dateutil.relativedelta import relativedelta
import copy

def load_all_data():
	f = open('raw_data.csv')
	lines = f.readlines()
	f.close()

	headers = lines[0].replace('\n', '').split(',')

	allData = []
	count = 0
	for i in range(1, len(lines)):
		line = lines[i].replace('\n', '').split(',')

		row = {}
		for i in range(len(line)):
			
			row[headers[i]] = line[i]
		
		allData.append(row)
		count += 1
		
	return allData

def convert_to_numpy_data(data, features):
	training_size = len(data) - 1

	X_train = np.zeros((training_size, len(features)))
	Y_train = np.zeros((training_size, 1))


	# sort by date
	data = sorted(data, key=lambda k: k['date'])

	for i in range(training_size):

		feature_count = 0
		for feature in data[i]:
			
			if feature in features:

				X_train[i][feature_count] = data[i][feature]
				feature_count += 1

		current_price = float(data[i]['PRC'])
		next_price = float(data[i + 1]['PRC'])

		predicted_label = 0
		if next_price > current_price:
			predicted_label = 1

		Y_train[i][0] = predicted_label

	return X_train, Y_train[:,0]

def filter_data(allData, ticker, start, end_year, num_days_to_check):
	data = []
	for row in allData:

		d = datetime.strptime(row['date'], '%Y%m%d')
	
		if row['TICKER'] == ticker and d >= start and d <= end_year and (d - start).days <= num_days_to_check: 
			data.append(row)
	
	return data


allData = load_all_data()

NUM_DAYS_IN_MONTH = 30
NUM_DAYS_IN_YEAR = 365

range_of_data = [10 * NUM_DAYS_IN_YEAR, 5 * NUM_DAYS_IN_YEAR, 1 * NUM_DAYS_IN_YEAR, 6 * NUM_DAYS_IN_MONTH, 3 * NUM_DAYS_IN_MONTH, 1 * NUM_DAYS_IN_MONTH, 15]
companies = ['MSFT', 'AAPL']
start_year = datetime.strptime('01/01/1989', '%m/%d/%Y')
end_year = datetime.strptime('12/31/2013', '%m/%d/%Y')
feature_combos = ['PRC', 'VOL', 'PRC,VOL'] # price, volume
NUM_YEARS_IN_SLIDING_WINDOW = 5

gamma = [.001, .01, .1, .5, 1, 10, 100]	
C = [.01, .1, 1, 10, 50, 100, 250, 500, 1000]

parameters = {'kernel':['rbf'], 'C':C, 'gamma':gamma}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)

for ticker in companies:

	for features in feature_combos:

		for num_days_to_check in range_of_data:

			start = copy.copy(start_year)

			while start < end_year: 
				data = filter_data(allData, ticker, start, end_year, num_days_to_check)

				if len(data) == 0:
					continue

				X_train, Y_train = convert_to_numpy_data(data, features.split(','))
				
				clf = clf.fit(X_train, Y_train)

				print '--------------------- NEW RUN ---------------------'
				#print clf.best_estimator_
				print 'Ticker: ', ticker
				print 'Features: ', features
				print 'Num days to check: ', num_days_to_check
				print 'Training size: ', len(X_train)
				print 'Start date: ', start
				print 'End date: ', end_year
				print 'Best params: ', clf.best_params_
				print 'Score: ', clf.best_score_
				print ''

				start += relativedelta(years=NUM_YEARS_IN_SLIDING_WINDOW) 

					






