import pandas
import numpy
from sklearn import svm

train = pandas.read_csv('training.csv')
labels = train[['Ca','P','pH','SOC','Sand']].values
test = pandas.read_csv('sorted_test.csv')

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

xtrain = numpy.array(train)[:695, :3594]
xval = numpy.array(train)[696:926, :3594]
xtest = numpy.array(train)[927:, :3594]

# Varuous values of C
models = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 
			90.0, 100.0];
SSEs, Cs = [], []

predictions = numpy.zeros((xval.shape[0], 5))
for i in range(len(models)):
	sup_vec = svm.SVR(C=models[i], verbose=2)
	for j in range(5):
		sup_vec.fit(xtrain, labels[:695, j])
		predictions[:, j] = sup_vec.predict(xval).astype(float)
	SSE = sum(sum((predictions - labels[696:926, :])**2))
	SSEs.append(SSE)
	Cs.append(models[i])

print 'cross validation...'
print 'C' + '\t' + 'SSE'
for i in range(len(models)):
	print str(Cs[i]) + '\t' + str(SSEs[i])

# C		SSE
# 1.0	423.783158424
# 10.0	360.839349289
# 20.0	341.239620849
# 30.0	329.842381563
# 40.0	322.855414415
# 50.0	317.038928654
# 60.0	313.114123444
# 70.0	311.949539366
# 80.0	308.455442542
# 90.0	305.941988269
# 100.0	304.855418421
# 100.0	304.855418421
# 150.0	297.610724264
# 200.0	294.240923582
# 250.0	290.135473808
# 300.0	285.857327203
# 350.0	284.4411996
# 400.0	281.606109822
# 450.0	277.671142624
# 500.0	276.403816841
# 550.0	276.896574895
# 600.0	276.935659491
# 650.0	276.40283349
# 700.0	277.118047019
# 750.0	278.356032264
# 800.0	280.008673547
# 850.0	281.290210469
# 900.0	282.151725115
# 950.0	282.829150324
# 1000.0	284.239502723

predictions = numpy.zeros((xtest.shape[0], 5))
sup_vec = svm.SVR(C=40.0, verbose=2)
for j in range(5):
	sup_vec.fit(numpy.array(train)[:926, :], labels[:926, j])
	predictions[:, j] = sup_vec.predict(xtest).astype(float)
SSE = sum(sum((predictions - labels[927:, :])**2))
print 'Based on C= ' + str(40.0) + ' Sum of squared errors (test set) is ' + str(SSE)

# 269.076066401 - C=650
# 266.014657485 - C=600
# 264.67075649 - C=550
# 262.849358977 - C=500
# 262.078083613 - C=450
# 261.40467797 - C=400
# 261.527901016 - C=350
# 260.05139962 - C=300
# 255.538892297 - C=250
# 239.54133357 - C=50
# 238.22611993 - C=40
# 239.596574327 - C=30
# 243.428028576 - C=20
# 253.489516953 - C=10
# 340.075077369 - C=1