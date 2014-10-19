## PLOT VALIDATION CURVE ##

import pandas
import numpy
from sklearn import svm
import pylab

train = pandas.read_csv('training.csv')
y = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)

xtrain = numpy.array(train)[:810, :3593]
xval = numpy.array(train)[811:-1, :3593]

# Varuous values of C
models = [70.0]

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

# based on 810 training examples and 3593 features

# C	train SSE	validation SSE
# 50.0	1511.5176484	512.769092011
# 60.0	1484.2642745	505.634627125
# 70.0	1459.86573946	501.68397213 --- BEST C
# 80.0	1439.03525943	503.383746689
# 90.0	1423.37436418	507.684806286
# 100.0	1411.85244097	509.933735849
# 110.0	1400.83312465	513.846776854
# 120.0	1385.05667742	515.910927813