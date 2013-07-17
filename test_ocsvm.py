import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from ocsvm import Ocsvm
from svdd import Svdd


if __name__ == '__main__':
	Dtrain = co.normal(2,100)
	svm = Ocsvm(Dtrain,0.1,'rbf',1.1)
	svm.train_dual()

	delta = 0.1
	x = np.arange(-4.0, 4.0, delta)
	y = np.arange(-4.0, 4.0, delta)
	X, Y = np.meshgrid(x, y)    
	(sx,sy) = X.shape
	Xf = np.reshape(X,(1,sx*sy))
	Yf = np.reshape(Y,(1,sx*sy))
	Dtest = np.append(Xf,Yf,axis=0)
	print(Dtest.shape)

	(res,state) = svm.apply_dual(co.matrix(Dtest))
	print(state)
	print(res.size)

	Z = np.reshape(res,(sx,sy))
	plt.contourf(X, Y, Z)
	plt.scatter(Dtrain[0,:],Dtrain[1,:],10)
	plt.show()

	print('finished')