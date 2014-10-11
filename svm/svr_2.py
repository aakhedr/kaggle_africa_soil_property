## PLOT VALIDATION CURVE ##

import pandas
import numpy
from sklearn import svm
import pylab

train = pandas.read_csv('training.csv')
y = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)

xtrain = numpy.array(train)[:695, :3593]
xval = numpy.array(train)[696:926, :3593]

# Varuous values of C
models = [300, 400, 500, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
gamma = .0001

trainSSEs, valSSEs = [], []

trainPredictions = numpy.zeros((xtrain.shape[0], 5))
valPredictions = numpy.zeros((xval.shape[0], 5))

for i in range(len(models)):
	sup_vec = svm.SVR(C=models[i], verbose=2, gamma=gamma)
	for j in range(5):
		sup_vec.fit(xtrain, y[:695, j])
		trainPredictions[:, j] = sup_vec.predict(xtrain).astype(float)
		valPredictions[:, j] = sup_vec.predict(xval).astype(float)

	trainSSE = sum(sum((trainPredictions - y[:695, :])**2))
	trainSSEs.append(trainSSE)

	valSSE = sum(sum((valPredictions - y[696:926, :])**2))
	valSSEs.append(valSSE)

print 'cross validation...'
print 'C' + '\t' + 'train SSE' + '\t' + 'validation SSE'
for i in range(len(models)):
	print str(models[i]) + '\t' + str(trainSSEs[i]) + '\t' + str(valSSEs[i])

pylab.plot(models, trainSSEs, label='error train')
pylab.plot(models, valSSEs, label='error validation')
pylab.title('Validation Curve with gamma={}'.format(gamma))
pylab.xlabel('Values of C')
pylab.ylabel('Sum of squared errors')
pylab.legend(loc='best')
pylab.show()