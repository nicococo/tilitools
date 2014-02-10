import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from ssvm import SSVM
from so_multiclass import SOMultiClass

if __name__ == '__main__':

	# generate raw training data
	Dtrain1 = co.normal(2,100)*0.2 + 2
	Dtrain2 = co.normal(2,100)*0.2 + 0
	Dtrain3 = co.normal(2,100)*0.2 - 2
	Dtrain = co.matrix([[Dtrain1], [Dtrain2], [Dtrain3]])
	Dtrain = co.matrix([[Dtrain.trans()],[co.matrix(1.0,(300,1))]]).trans()
	Dy = co.matrix([[co.matrix(0,(1,100))], [co.matrix(1,(1,100))], [co.matrix(2,(1,100))]])

	# generate structured object
	sobj = SOMultiClass(Dtrain,Dy)

	# train svdd
	ssvm = SSVM(sobj,1.0)
	(ws,slacks) = ssvm.train()
	print(ws)
#	print(slacks)

	# generate test data grid
	delta = 0.1
	x = np.arange(-4.0, 8.0, delta)
	y = np.arange(-4.0, 8.0, delta)
	X, Y = np.meshgrid(x, y)    
	(sx,sy) = X.shape
	Xf = np.reshape(X,(1,sx*sy))
	Yf = np.reshape(Y,(1,sx*sy))
	Dtest = np.append(Xf,Yf,axis=0)
	Dtest = np.append(Dtest,np.reshape([1.0]*(sx*sy),(1,sx*sy)),axis=0)
	print(Dtest.shape)

	# generate structured object
	predsobj = SOMultiClass(co.matrix(Dtest),co.matrix([3]))

	(res,cls,msg) = ssvm.apply(predsobj)
	print(res.size)

	# nice visualization
	Z = np.reshape(cls,(sx,sy))
	plt.contourf(X, Y, Z)
	plt.scatter(Dtrain[0,:],Dtrain[1,:],10)
	plt.show()

	print('finished')