import numpy as np
import pylab as pl
from bdd import BDD
from kernel import Kernel
import cvxopt as co


if __name__ == '__main__':
	kparam = 2.1
	ktype = 'rbf'

	# generate raw training data
	Dtrain1 = co.normal(2,100)*0.2
	Dtrain2 = co.normal(2,100)*0.3 + 0.8
	Dtrain = co.matrix([Dtrain1.trans(), Dtrain2.trans()]).trans()
	# build kernel
	kernel = Kernel.get_kernel(Dtrain,Dtrain,ktype,kparam)
	# train bdd
	bdd = BDD(kernel)
	# print bdd
	bdd.train_BDD()


	# generate test data grid
	delta = 0.1
	x = np.arange(-4.0, 4.0, delta)
	y = np.arange(-4.0, 4.0, delta)
	X, Y = np.meshgrid(x, y)    
	(sx,sy) = X.shape
	Xf = np.reshape(X,(1,sx*sy))
	Yf = np.reshape(Y,(1,sx*sy))
	Dtest = np.append(Xf,Yf,axis=0)
	print(Dtest.shape)

	# build kernel map	
	kernel_map = Kernel.get_kernel(co.matrix(Dtest),Dtrain,ktype,kparam)
	# build the diagonal of the test kernel
	norms = Kernel.get_diag_kernel(co.matrix(Dtest),ktype,kparam)
	scores, mat = bdd.apply_BDD(Dtest, kernel_map, norms)
	Z = np.reshape(scores,(sx,sy))
	pl.contourf(X, Y, Z)
	pl.scatter(Dtrain[0,:],Dtrain[1,:],10)
	pl.show()