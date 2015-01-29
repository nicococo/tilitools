import cvxopt as co
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from ssvm import SSVM
from so_multiclass import SOMultiClass

class ToyData:


    @staticmethod
    def get_gaussian(num,dims=2,means=[0,0],vars=[1,1]):
        data = co.matrix(0.0,(dims,num))
        for d in range(dims):
            data[d,:] = co.normal(1,num)*vars[d] + means[d]
        return data


    @staticmethod
    def get_2state_gaussian_seq(lens,dims=2,means1=[2,2,2,2],means2=[5,5,5,5],vars1=[1,1,1,1],vars2=[1,1,1,1],anom_prob=1.0):
        
        seqs = co.matrix(0.0, (dims, lens))
        lbls = co.matrix(0, (1,lens))
        marker = 0

        # generate first state sequence
        for d in range(dims):
            seqs[d,:] = co.normal(1,lens)*vars1[d] + means1[d]

        prob = np.random.uniform()
        if prob<anom_prob:      
            # add second state blocks
            while (True):
                max_block_len = 0.6*lens
                min_block_len = 0.1*lens
                block_len = np.int(max_block_len*np.single(co.uniform(1))+3)
                block_start = np.int(lens*np.single(co.uniform(1)))

                if (block_len - (block_start+block_len-lens)-3>min_block_len):
                    break

            block_len = min(block_len,block_len - (block_start+block_len-lens)-3)
            lbls[block_start:block_start+block_len-1] = 1
            marker = 1
            for d in range(dims):
                #print block_len
                seqs[d,block_start:block_start+block_len-1] = co.normal(1,block_len-1)*vars2[d] + means2[d]

        return (seqs, lbls, marker)



    @staticmethod
    def get_2state_anom_seq(lens, comb_block_len, anom_prob=1.0, num_blocks=1):
        
        seqs = co.matrix(0.0, (1, lens))
        lbls = co.matrix(0, (1, lens), tc='i')
        marker = 1

        # generate first state sequence, gaussian noise 0=mean, 1=variance
        seqs[0,:] = co.normal(1, lens)*1.0
        bak = co.matrix(seqs)
        
        prob = np.random.uniform()
        if prob<anom_prob:      

            marker = 0

            # add a single block
            blen = 0
            for b in xrange(np.int(num_blocks)):
                if (b==num_blocks-1 and b>1):
                    block_len = np.round(comb_block_len-blen)
                else:
                    # add second state blocks
                    block_len = np.int(np.floor((comb_block_len-blen)/float(num_blocks-b)))
                
                bcnt = 0
                isDone = False
                while isDone==False and bcnt<100000:
                    bcnt += 1
                    start = np.random.randint(low=0,high=lens-block_len+1)
                    if (sum(lbls[0,start:start+block_len])==0):
                        #print start
                        #print block_len
                        #print start+block_len
                        lbls[0,start:start+block_len] = 1
                        seqs[0,start:start+block_len] = (bak[0,start:start+block_len]+0.6)
                        isDone = True
                        break
                if isDone:
                    blen += block_len
            print('Anomamly block lengths (target/reality)= {0}/{1} '.format(comb_block_len, blen))
            if not comb_block_len==blen:
                print('Warning! Anomamly block length error.')

        return (seqs, lbls, marker)
