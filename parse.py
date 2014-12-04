
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
import ast
import csv
from sklearn import svm, grid_search
from sklearn.metrics import accuracy_score, recall_score, precision_score
from nn import NeuralNet

def percentChange(startPoint, currentPoint):
	realmin = 0.00000000001
	try:
		x = ((float(currentPoint)-float(startPoint))/float(startPoint))*100.0
		if x == 0.0:
			return realmin
		else:
			return x
	except:
		return realmin


def patternStorage(ticker, prop):
    # patStartTime = time.time()

    avgLine = dow30[ticker]

    x = len(avgLine)-(int(numPointsInPattern)*2)

    y = int(numPointsInPattern)+1

    while y < x:
        pattern = []
        patternPr = []
        for i in reversed(range(0,int(numPointsInPattern))):
            pattern.append(percentChange(avgLine[y-int(numPointsInPattern)][prop], avgLine[y-i][prop]))
            # pattern.append(percentChange(avgLine[y-int(numPointsInPattern)]['volume'], avgLine[y-i]['volume']))


        for j in reversed(range(0,int(numPointsInPattern))):
             patternPr.append(avgLine[y-j]['price'])

        patternPrice.append(patternPr)

        fraction_outcome = int(numPointsInPattern * 0.1)
        if fraction_outcome < 2:
            fraction_outcome = 2
        outcomeRange = avgLine[y+1:y+fraction_outcome]
        # outcomeRange = avgLine[y+1:y+3]
        # outcomeRange = avgLine[y+20:y+30]
        currentPoint = avgLine[y][prop]

        totalOutcome = 0
        for a in outcomeRange:
            totalOutcome += a['price']

        avgOutcome = totalOutcome / len(outcomeRange)

        futureOutcome = percentChange(currentPoint, avgOutcome)
        # futureOutcome = int(np.around(futureOutcome, decimals=0))
        # futureOutcome = int(np.around(avgOutcome, decimals=0))

        # discretize outcome
        # print futureOutcome
        truePercentChg.append(futureOutcome)

        if futureOutcome > 1:
            futureOutcome = 1
        elif futureOutcome < -1:
            futureOutcome = -1
        else:
            futureOutcome = 0

        patternAr.append(pattern)
        performanceAr.append(futureOutcome)
        # print avgOutcome
        performancePrice.append(avgOutcome)

        y += 1

	# patEndTime = time.time()




filename = 'dow30_2013-1990.csv'

dow30 = {}


with open(filename, 'rb') as csv_file:
    tickerReader = csv.reader(csv_file, delimiter=',')
    r = 0
    for row in tickerReader:
        if r != 0:
            if row[2] not in dow30:
                print row[2]
                dow30[row[2]] = []
                # last = row[2]

            if '.' in row[4]:
                prc = float(row[4])
            else:
                continue
            if row[5].isdigit():
                vol = int(row[5])
            else:
                continue
            dow30[row[2]].append( {'date':row[1], 'price':prc, 'volume':vol, 'pv': prc*vol } )

        r += 1




numPointsInPattern = 21
patternAr = []
performanceAr = []
patternPrice = []
performancePrice = []
truePercentChg = []

mfield = 'price'

# for ticker in dow30:
#     patternAr = []
#     performanceAr = []
#     patternPrice = []
#     performancePrice = []
#     truePercentChg = []
#
#     patternStorage(ticker, mfield)
#     n = int(len(patternAr) * 0.5)
#     patternAr = np.array(patternAr)
#     performanceAr = np.array(performanceAr)
#     model = svm.SVC(kernel='rbf', C=1.1, gamma=0.011)
#     model.fit(patternAr[0:n], performanceAr[0:n])
#
#     n = len(patternAr) - 100
#
#     ypred = model.predict(patternAr[n:])
#     accuracy = accuracy_score(performanceAr[n:], ypred)
#     precision = precision_score(performanceAr[n:], ypred)
#     recall = recall_score(performanceAr[n:], ypred)
#
#     print ticker,"Accuracy = "+str(accuracy)
#     print ticker,"Precision = "+str(precision)
#     print ticker,"Recall = "+str(recall)
#
#     result = performanceAr[n:]
#     patternResult = patternAr[n:]
#     prices = performancePrice[n:]
#     percent = truePercentChg[n:]
#     successfulTrades = 0
#     profit = prices[0]
#     current_percent = 0
#
#     for i in range(len(ypred)):
#         if result[i] == ypred[i]:
#             profit += profit * abs(percent[i]/100.0)
#             successfulTrades += 1
#
#     print 'Total Return:',(profit - prices[0])
#     print 'Percent return:',((profit-prices[0])/prices[0])*100
#     print 'Successful trades:',successfulTrades


for ticker in dow30:
    patternStorage(ticker, mfield)

# patternStorage('NKE', mfield)
# patternStorage('DIS', mfield)
# patternStorage('MMM', mfield)
# patternStorage('UNH', mfield)
# patternStorage('INTC', mfield)
# patternStorage('HD', mfield)
# patternStorage('GS', mfield)
# patternStorage('JPM', mfield)

# print 'output nodes:',len(np.unique(performanceAr))

n = int(len(patternAr) * 0.5)
# print n

patternAr = np.array(patternAr)

# print patternAr[1:4]

performanceAr = np.array(performanceAr)


# gamma = np.arange(0.001,0.1,0.005)  # [.001, .01, .1, .5, 1, 10, 100]
# C = np.arange(0.1,10,0.5)  # [.01, .1, 1, 10, 50, 100, 250, 500, 1000]
#
# parameters = {'kernel':['rbf'], 'C':C, 'gamma':gamma}
# svr = svm.SVC()
# #model = grid_search.GridSearchCV(svr, parameters)
model = svm.SVC(kernel='rbf', C=1.1, gamma=0.011)
# # model = model.fit(X_train, Y_train)
# #model = NeuralNet(np.array([43,41,43]), .70, 0.0001, 100)  # 100 @ 2.5 = 0.885, 400 @ 1.6 = 0.88, 1000 @ 1 = 0.8542,
model.fit(patternAr[0:n], performanceAr[0:n])

# patternAr = []
# performanceAr = []
#
# patternStorage('JPM', 'price')
#
# patternAr = np.array(patternAr)
# performanceAr = np.array(performanceAr)

n = len(patternAr) - 100
# n += 200

ypred = model.predict(patternAr[n:])

#print 'Best params: ', model.best_params_
#print 'Score: ', model.best_score_

# print ypred
# print performanceAr[n:]
#
accuracy = accuracy_score(performanceAr[n:], ypred)
precision = precision_score(performanceAr[n:], ypred)
recall = recall_score(performanceAr[n:], ypred)

print "NeuralNet Accuracy = "+str(accuracy)
print "NeuralNet Precision = "+str(precision)
print "NeuralNet Recall = "+str(recall)

# model.visualizeHiddenNodes('hiddenLayers.png')

# fig = plt.figure(figsize=(10,6))
# plt.ion()

# xp = range(0,int(numPointsInPattern))
result = performanceAr[n:]
patternResult = patternAr[n:]
prices = performancePrice[n:]
percent = truePercentChg[n:]
successfulTrades = 0
profit = prices[0]
current_percent = 0

for i in range(len(ypred)):
    if result[i] == ypred[i]:
        profit += profit * abs(percent[i]/100.0)
        successfulTrades += 1
    # print profit

    # if result[i] == ypred[i]:
    #     # print patternResult[i]
    #     # print performanceAr[i]
    #     plt.figure(successfulTrades)
    #     successfulTrades += 1
    #     plt.scatter(numPointsInPattern+5, result[i], s=25)
    #     plt.subplot(1,2,1)
    #     plt.plot(xp, patternPrice[i], linewidth=1)
    #     plt.grid(True)
    #     plt.subplot(1,2,2)
    #     plt.plot(xp, patternResult[i], linewidth=1)
    #     plt.grid(True)
    #     plt.draw()
        # if successfulTrades > 0:
        #     break

print 'Total Return:',(profit - prices[0])
print 'Percent return:',((profit-prices[0])/prices[0])*100

print 'Successful trades:',successfulTrades

# plt.show()