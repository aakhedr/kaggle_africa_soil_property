## PLOT VALIDATION CURVE ##

import pandas
import numpy
from sklearn import svm
import pylab

train = pandas.read_csv('training.csv')
y = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)

xtrain = numpy.array(train)[:810, :3578]
xval = numpy.array(train)[811:-1, :3578]

# Varuous values of C
models = [90.0]

trainSSEs, valSSEs = [], []

trainPredictions = numpy.zeros((xtrain.shape[0], 5))
valPredictions = numpy.zeros((xval.shape[0], 5))

for i in range(len(models)):
	sup_vec = svm.SVR(C=models[i], verbose=2)
	for j in range(5):
		sup_vec.fit(xtrain, y[:810, j])
		trainPredictions[:, j] = sup_vec.predict(xtrain).astype(float)
		valPredictions[:, j] = sup_vec.predict(xval).astype(float)

	trainSSE = sum(sum((trainPredictions - y[:810, :])**2))
	trainSSEs.append(trainSSE)

	valSSE = sum(sum((valPredictions - y[811:-1, :])**2))
	valSSEs.append(valSSE)

print 'cross validation...'
print 'C' + '\t' + 'train SSE' + '\t' + 'validation SSE'
for i in range(len(models)):
	print str(models[i]) + '\t' + str(trainSSEs[i]) + '\t' + str(valSSEs[i])

pylab.plot(models, trainSSEs, 'ro', label='error train')
pylab.plot(models, valSSEs, 'bx', label='error validation')
pylab.title('Validation Curve with gamma={}'.format('default'))
pylab.xlabel('Values of C')
pylab.ylabel('Sum of squared errors')
pylab.legend(loc='best')
pylab.show()