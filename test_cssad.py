import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from cssad import Cssad
from ocsvm import Ocsvm


if __name__ == '__main__':
	# example constants (training set size and splitting)
	N_pos = 50
	N_neg = 50
	N_unl = 500

	# generate training labels
	yp = co.matrix(1,(1,N_pos),'i')
	yu = co.matrix(0,(1,N_unl),'i')
	yn = co.matrix(-1,(1,N_neg),'i')
	Dy = co.matrix([[yp], [yu], [yn]])
	
	# generate training data
	Dtrainp = co.normal(2,N_pos)*0.5
	Dtrainu = co.normal(2,N_unl)*0.5
	Dtrainn = co.normal(2,N_neg)*0.3+1.5
	Dtrain = co.matrix([[Dtrainp], [Dtrainu], [Dtrainn]])

	# train convex semi-supervised anomaly detection
	svm = Cssad(Dtrain,Dy,1.01,1.0,1.0,1.0,'rbf',0.1)
	#svm = Ocsvm(Dtrain,0.1,'rbf',1.1)
	svm.train_dual()

	# generate test data from a grid for nicer plots
	delta = 0.1
	x = np.arange(-4.0, 4.0, delta)
	y = np.arange(-4.0, 4.0, delta)
	X, Y = np.meshgrid(x, y)    
	(sx,sy) = X.shape
	Xf = np.reshape(X,(1,sx*sy))
	Yf = np.reshape(Y,(1,sx*sy))
	Dtest = np.append(Xf,Yf,axis=0)
	print(Dtest.shape)

	# predict 	
	(res,state) = svm.apply_dual(co.matrix(Dtest))

	# make a nice plot of it
	Z = np.reshape(res,(sx,sy))
	plt.contourf(X, Y, Z)
	plt.contour(X, Y, Z, [np.array(svm.get_threshold())[0,0]])

	plt.scatter(Dtrain[0,svm.get_support_dual()],Dtrain[1,svm.get_support_dual()],40,c='k') 

	plt.scatter(Dtrain[0,N_pos:N_pos+N_unl-1],Dtrain[1,N_pos:N_pos+N_unl-1],10,c='g') 
	plt.scatter(Dtrain[0,0:N_pos],Dtrain[1,0:N_pos],20,c='r') 
	plt.scatter(Dtrain[0,N_pos+N_unl:],Dtrain[1,N_pos+N_unl:],20,c='b') 

	#plt.scatter(Dtrain[0,svm.get_support_dual()],Dtrain[1,svm.get_support_dual()],40,c='k') 
	plt.show()

	print('finished')