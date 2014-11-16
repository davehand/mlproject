
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time



def percentChange(startPoint, currentPoint):
	realmin = 0.00000000001
	try:
		x = ((float(currentPoint)-startPoint)/abs(startPoint))*100.0
		if x == 0.0:
			return realmin
		else:
			return x
	except:
		return realmin


def patternStorage():
	# patStartTime = time.time()

	x = len(avgLine)-(int(numPointsInPattern)*2)

	y = int(numPointsInPattern)+1
	while y < x:
		pattern = []
		for i in reversed(range(0,int(numPointsInPattern))):
			pattern.append(percentChange(avgLine[y-int(numPointsInPattern)], avgLine[y-i]))


		outcomeRange = avgLine[y+20:y+30]
		currentPoint = avgLine[y]

		try:
			avgOutcome = reduce(lambda x, y: x+y, outcomeRange) / len(outcomeRange)
		except Exception, e:
			print str(e)
			avgOutcome=0

		futureOutcome = percentChange(currentPoint, avgOutcome)
		
		patternAr.append(pattern)
		performanceAr.append(futureOutcome)

		y += 1
		
	# patEndTime = time.time()



def currentPattern():
	num = int(numPointsInPattern)+1
	for i in reversed(range(1,num)):
		cp = percentChange(avgLine[-num], avgLine[-i])
		patForRec.append(cp)

	# print patForRec

def patternRecognition():
	
	predictedOutcomesAr = []
	global patFound
	patFound = 0
	plotPatAr = []

	plt.clf()

	for eachPattern in patternAr[:-5]:
		
		simSum = 0
		for i in range(0,int(numPointsInPattern)):
			sim = 100.00 - abs(percentChange(eachPattern[i], patForRec[i]))
			if sim < 50:
				break
			simSum += sim

		howSim = simSum / numPointsInPattern
		
		if howSim > similarityThreshold:
			patdex = patternAr.index(eachPattern)
			patFound = 1
			plotPatAr.append(eachPattern)


	predArray = []

	if patFound == 1:
		xp = range(0,int(numPointsInPattern))
		# fig = plt.figure(figsize=(10,6))
		lastIdx = int(numPointsInPattern)-1

		for eachPatt in plotPatAr:
			futurePoints = patternAr.index(eachPatt)

			if performanceAr[futurePoints] > patForRec[lastIdx]:
				pcolor = '#00cc00'
				predArray.append(1.000)

			else:
				pcolor = '#d44000'
				predArray.append(-1.000)

			plt.plot(xp, eachPatt)
			predictedOutcomesAr.append(performanceAr[futurePoints])
			plt.scatter(lastIdx+5, performanceAr[futurePoints], c=pcolor,alpha=.3)

		realOutcomeRange = allData[toWhat+20:toWhat+30]
		realAvgOutcome = reduce(lambda x, y: x+y, realOutcomeRange) / len(realOutcomeRange)
		realMovement = percentChange(allData[toWhat], realAvgOutcome)
		predictedAvgOutcome = reduce(lambda x, y: x+y, predictedOutcomesAr) / len(predictedOutcomesAr)
		
		# print predArray
		predictionAverage = reduce(lambda x, y: x+y, predArray) / len(predArray)
		# print predictionAverage
		if predictionAverage < 0:
			print 'drop predicted'
			print patForRec[lastIdx]
			print realMovement
			if realMovement < patForRec[lastIdx]:
				accuracyArray.append(100)
			else:
				accuracyArray.append(0)

		if predictionAverage > 0:
			print 'rise predicted'
			print patForRec[lastIdx]
			print realMovement
			if realMovement > patForRec[lastIdx]:
				accuracyArray.append(100)
			else:
				accuracyArray.append(0)

		plt.scatter(lastIdx+10, realMovement, c='#54fff7', s=25)
		plt.scatter(lastIdx+10, predictedAvgOutcome, c='b', s=25)

		plt.plot(xp, patForRec, '#54fff7', linewidth=3)
		plt.grid(True)
		plt.title('Pattern Recognition')
		plt.draw()



# def graphRawFX():
# 	'''
# 		plot raw forex data
# 	'''

# 	fig = plt.figure(figsize=(10,7))
# 	ax1 = plt.subplot2grid((40,40), (0,0), rowspan=40, colspan=40)

# 	ax1.plot(date,bid)
# 	ax1.plot(date,ask)

# 	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
# 	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
# 	for label in ax1.xaxis.get_ticklabels():
# 		label.set_rotation(45)

# 	ax1_2 = ax1.twinx()
# 	ax1_2.fill_between(date, 0, (ask-bid), facecolor='g', alpha=.3)
# 	plt.subplots_adjust(bottom=.23)

# 	plt.grid(True)
# 	plt.show()



totalStart = time.time()

date, bid, ask = np.loadtxt('GBPUSD1d.txt', unpack=True, delimiter=',',
							converters={0:mdates.strpdate2num('%Y%m%d%H%M%S')})


dataLength = int(bid.shape[0])
print 'data length is',dataLength

toWhat = 3700
allData = ((bid+ask)/2)

accuracyArray = []
samps = 0

numPointsInPattern = 30.00
similarityThreshold = 70


fig = plt.figure(figsize=(10,6))
plt.ion()
plt.show()

# Problem as data gets large
# patternAr = []
# performanceAr = []
# patForRec = []

while toWhat < dataLength:

	# avgLine = ((bid+ask)/2)
	avgLine = allData[:toWhat]

	# Problem as data gets large
	patternAr = []
	performanceAr = []
	patForRec = []

	# print 'Starting processing ...'
	patternStorage()
	currentPattern()
	patternRecognition()
	totalTime = time.time() - totalStart 

	# print 'Total processing time took:',totalTime,'seconds'
	# moveOn = raw_input('press ENTER to continue...')
	samps += 1
	toWhat += 1
	if len(accuracyArray) > 0:
		accuracyAverage = reduce(lambda x, y: x+y, accuracyArray) / len(accuracyArray)
		print 'Backtested Accuracy is',str(accuracyAverage)+'% after',samps,'samples'

