import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from ssvm import SSVM
from latent_svdd import LatentSVDD
from structured_ocsvm import StructuredOCSVM
from structured_pca import StructuredPCA
from so_multiclass import SOMultiClass
from utils_data import ToyData


if __name__ == '__main__':
	NUM_CLASSES = 6

	# generate raw training data
	Dtrain1 = ToyData.get_gaussian(100,dims=2,means=[1.0,1.0],vars=[1.0,0.3])
	Dtrain2 = ToyData.get_gaussian(10,dims=2,means=[-2.0,1.0],vars=[0.3,1.3])
	Dtrain3 = ToyData.get_gaussian(10,dims=2,means=[3.0,-3.0],vars=[0.3,0.3])
	Dtrain4 = ToyData.get_gaussian(5,dims=2,means=[6.0,-3.0],vars=[0.2,0.1])

	Dtrain = co.matrix([[Dtrain1], [Dtrain2], [Dtrain3], [Dtrain4]])
	Dtrain = co.matrix([[Dtrain.trans()],[co.matrix(1.0,(125,1))]]).trans()
	Dy = co.matrix([co.matrix([0]*100), co.matrix([1]*10), co.matrix([2]*10), co.matrix([3]*5)])

	# generate structured object
	sobj = SOMultiClass(Dtrain, y=Dy , classes=NUM_CLASSES)

	# unsupervised methods
	lsvdd = LatentSVDD(sobj,1.0/(125.0*1.0))
	spca = StructuredPCA(sobj)
	socsvm = StructuredOCSVM(sobj,1.0/(125.0*1.0))
	# supervised methods
	ssvm = SSVM(sobj) 

	# generate test data grid
	delta = 0.2
	x = np.arange(-8.0, 8.0, delta)
	y = np.arange(-8.0, 8.0, delta)
	X, Y = np.meshgrid(x, y)    
	(sx,sy) = X.shape
	Xf = np.reshape(X,(1,sx*sy))
	Yf = np.reshape(Y,(1,sx*sy))
	Dtest = np.append(Xf,Yf,axis=0)
	Dtest = np.append(Dtest,np.reshape([1.0]*(sx*sy),(1,sx*sy)),axis=0)
	print(Dtest.shape)

	# generate structured object
	predsobj = SOMultiClass(co.matrix(Dtest),NUM_CLASSES)


	# for all methods
	fig = plt.figure()
	for i in xrange(4):
		plt.subplot(2,4,i+1)
		
		if i==0:
			plt.title("LatentSVDD")
			lsvdd.train_dc()
			(scores,lats) = lsvdd.apply(predsobj)
		if i==1:
			plt.title("StructPCA")
			spca.train_dc()
			(scores,lats) = spca.apply(predsobj)
		if i==2:
			plt.title("StructOCSVM")
			socsvm.train_dc()
			(scores,lats) = socsvm.apply(predsobj)
		if i==3:
			plt.title("SSVM")
			ssvm.train()
			(scores,lats) = ssvm.apply(predsobj)

		# plot scores
		Z = np.reshape(scores,(sx,sy))
		plt.contourf(X, Y, Z)
		plt.scatter(Dtrain[0,:],Dtrain[1,:],10)

		# plot latent variable
		Z = np.reshape(lats,(sx,sy))
		plt.subplot(2,4,i+4+1)
		plt.contourf(X, Y, Z)
		plt.scatter(Dtrain[0,:],Dtrain[1,:],10)

	plt.show()

	print('finished')