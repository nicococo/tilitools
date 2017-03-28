from cvxopt import matrix,spmatrix,sparse,mul,spdiag
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
from kernel import Kernel  

class SSAD:
    """Convex semi-supervised anomaly detection with hinge-loss and L2 regularizer
        as described in Goernitz et al., Towards Supervised Anomaly Detection, JAIR, 2013

               minimize 			0.5 ||w||^2_2 - rho - kappa*gamma
        {w,rho,gamma>=0,xi>=0} 		+ eta_u sum_i xi_i + eta_l sum_j xi_j
        subject to   <w,phi(x_i)> >= rho - xi_i
                  y_j<w,phi(x_j)> >= y_j*rho + gamma - xi_j

        And the corresponding dual optimization problem:

            maximize -0.5 sum_(i,j) alpha_i alpha_j y_i y_j k(x_i,x_j)
        {0<=alpha_i<=eta_i}
        subject to 	kappa <= sum_j alpha_j  (for all labeled examples)

        For better readability, we also introduce labels y_i=+1 for all unlabeled
        examples which enables us to combine sums.

        Note: Only dual solution is supported.

        Written by: Nico Goernitz, TU Berlin, 2013/14
    """

    MSG_ERROR = -1	# (scalar) something went wrong
    MSG_OK = 0	# (scalar) everything alright

    PRECISION = 10**-5 # important: effects the threshold, support vectors and speed!

    cy = [] # (vector) converted label vector (+1 for pos and unlabeled, -1 for outliers)
    cl = [] # (vector) converted label vector (+1 for labeled examples, 0.0 for unlabeled)

    samples = -1 	# (scalar) amount of training data in X
    labeled = -1 	# (scalar) amount of labeled data

    cC = [] # (vector) converted upper bound box constraint for each example
    Cp = 1.0	# (scalar) the regularization constant for positively labeled samples > 0
    Cu = 1.0    # (scalar) the regularization constant for unlabeled samples > 0
    Cn = 1.0    # (scalar) the regularization constant for outliers > 0

    kappa = 1.0 # (scalar) regularizer for importance of the margin

    kernel = []	# (matrix) kernel matrix
    y = []	# (vector) corresponding labels (+1,-1 and 0 for unlabeled)

    alphas = []	# (vector) dual solution vector
    svs = []	# (vector) list of support vector (contains indices)

    threshold = matrix(0.0)	# (scalar) the optimized threshold (rho)



    def __init__(self, kernel, y, kappa=1.0, Cp=1.0, Cu=1.0, Cn=1.0):
        """ Constructor """
        self.kernel = kernel
        self.y = y
        self.kappa = kappa
        self.Cp = Cp
        self.Cu = Cu
        self.Cn = Cn
        (foo,self.samples) = y.size

        # these are counters for pos,neg and unlabeled examples
        npos = nunl = nneg = 0
        # these vectors are used in the dual optimization algorithm
        self.cy = matrix(1.0,(1,self.samples)) # cy=+1.0 (unlabeled,pos) & cy=-1.0 (neg)
        self.cl = matrix(1.0,(1,self.samples)) # cl=+1.0 (labeled) cl=0.0 (unlabeled)
        self.cC = matrix(Cp,(self.samples,1)) # cC=Cu (unlabeled) cC=Cp (pos) cC=Cn (neg)
        for i in range(self.samples):
            if y[0,i]==0:
                nunl += 1
                self.cl[0,i] = 0.0
                self.cC[i,0] = Cu
            if y[0,i]==-1:
                nneg += 1
                self.cy[0,i] = -1.0
                self.cC[i,0] = Cn
        npos = self.samples - nneg - nunl
        self.labeled = npos+nneg

        # if there are no labeled examples, then set kappa to 0.0 otherwise
        # the dual constraint kappa <= sum_{i \in labeled} alpha_i = 0.0 will
        # prohibit a solution
        if nunl==self.samples:
            print('There are no labeled examples hence, setting kappa=0.0')
            self.kappa = 0.0

        print('Creating new convex semi-supervised anomaly detection with {0} samples.'.format(self.samples))
        print('There are {0} positive, {1} unlabeled and {2} outlier examples.'.format(npos,nunl,nneg))


    def set_train_kernel(self,kernel):
        (dim1,dim2) = kernel.size
        if (dim1!=dim2 and dim1!=self.samples):
            print('(Kernel) Wrong format.')
            return SSAD.MSG_ERROR
        self.kernel = kernel;
        return SSAD.MSG_OK


    def train_dual(self):
        """Trains an one-class svm in dual with kernel."""
        if (self.samples<1):
            print('Invalid training data.')
            return SSAD.MSG_ERROR

        # number of training examples
        N = self.samples

        # generate the label kernel
        Y = self.cy.trans()*self.cy

        # generate the final PDS kernel
        P = mul(self.kernel,Y)

        # check for PDS
        eigs = np.linalg.eigvalsh(np.array(P))
        if (eigs[0]<0.0):
            print('Smallest eigenvalue is {0}'.format(eigs[0]))
            P += spdiag([-eigs[0] for i in range(N)])
            print(np.linalg.eigvalsh(np.array(P)))

        # there is no linear part of the objective
        q = matrix(0.0, (N,1))

        # sum_i y_i alpha_i = A alpha = b = 1.0
        A = self.cy
        b = matrix(1.0, (1,1))

        # inequality constraints: G alpha <= h
        # 1) alpha_i  <= C_i
        # 2) -alpha_i <= 0
        # 3) kappa <= \sum_i labeled_i alpha_i  -> -cl' alpha <= -kappa
        G1 = spmatrix(1.0, range(N), range(N))
        G3 = -self.cl
        h1 = self.cC
        h2 = matrix(0.0, (N,1))
        h3 = -self.kappa

        G  = sparse([G1,-G1])
        h  = matrix([h1,h2])
        if (self.labeled>0):
            print('Labeled data found.')
            G  = sparse([G1,-G1,G3])
            h  = matrix([h1,h2,h3])

        # solve the quadratic programm
        sol = qp(P,-q,G,h,A,b)

        # store solution
        self.alphas = sol['x']

        # 1. find all support vectors, i.e. 0 < alpha_i <= C
        # 2. store all support vector with alpha_i < C in 'margins'
        self.svs = []
        for i in range(N):
            if (self.alphas[i]>SSAD.PRECISION):
                self.svs.append(i)

        # these should sum to one
        print('Validate solution:')
        print('- found {0} support vectors'.format(len(self.svs)))

        summe = 0.0
        for i in range(N): summe += self.alphas[i]*self.cy[0,i]
        print('- sum_(i) alpha_i cy_i = {0} = 1.0'.format(summe))

        summe = 0.0
        for i in self.svs: summe += self.alphas[i]*self.cy[0,i]
        print('- sum_(i in sv) alpha_i cy_i = {0} ~ 1.0 (approx error)'.format(summe))

        summe = 0.0
        for i in range(N): summe += self.alphas[i]*self.cl[0,i]
        print('- sum_(i in labeled) alpha_i = {0} >= {1} = kappa'.format(summe,self.kappa))

        summe = 0.0
        for i in range(N): summe += self.alphas[i]*(1.0-self.cl[0,i])
        print('- sum_(i in unlabeled) alpha_i = {0}'.format(summe))
        summe = 0.0
        for i in range(N):
            if (self.y[0,i]>=1.0): summe += self.alphas[i]
        print('- sum_(i in positives) alpha_i = {0}'.format(summe))
        summe = 0.0
        for i in range(N):
            if (self.y[0,i]<=-1.0): summe += self.alphas[i]
        print('- sum_(i in negatives) alpha_i = {0}'.format(summe))

        # infer threshold (rho)
        self.calculate_threshold_dual()

        (thres, MSG) = self.apply_dual(self.kernel[self.svs,self.svs])
        T = np.array(self.threshold)[0,0]
        cnt = 0
        for i in range(len(self.svs)):
            if thres[i,0]<(T-SSAD.PRECISION):
                cnt += 1
        print('Found {0} support vectors. {1} of them are outliers.'.format(len(self.svs),cnt))

        return SSAD.MSG_OK


    def calculate_threshold_dual(self):
        # 1. find all support vectors, i.e. 0 < alpha_i <= C
        # 2. store all support vector with alpha_i < C in 'margins'
        margins = []
        for i in self.svs:
            if (self.alphas[i]<(self.cC[i,0]-SSAD.PRECISION)):
                margins.append(i)

        # 3. infer threshold from examples that have 0 < alpha_i < Cx
        # where labeled examples lie on the margin and
        # unlabeled examples on the threshold

        # build the test kernel
        (thres,MSG) = self.apply_dual(self.kernel[margins,self.get_support_dual()])

        idx = 0
        pos = neg = 0.0
        flag = flag_p = flag_n = False
        for i in margins:
            if (self.y[0,i]==0):
                # unlabeled examples with 0<alpha<Cu lie direct on the hyperplane
                # hence, it is sufficient to use a single example
                self.threshold = thres[idx,0]
                flag = True
                break
            if (self.y[0,i]>=1):
                pos = thres[idx,0]
                flag_p = True
            if (self.y[0,i]<=-1):
                neg = thres[idx,0]
                flag_n = True
            idx += 1

        # check all possibilities (if no unlabeled examples with 0<alpha<Cu was found):
        # a) we have a positive and a negative, then the threshold is in the middle
        if (flag==False and flag_n==True and flag_p==True):
            self.threshold = 0.5*(pos+neg)
        # b) there is only a negative example, approx with looser threshold
        if (flag==False and flag_n==True and flag_p==False):
            self.threshold = neg-SSAD.PRECISION
        # c) there is only a positive example, approx with tighter threshold
        if (flag==False and flag_n==False and flag_p==True):
            self.threshold = pos+SSAD.PRECISION

        # d) no pos,neg or unlabeled example with 0<alpha<Cx found :(
        if (flag==flag_p==flag_n==False):
            print('Poor you, guessing threshold...')
            (thres,MSG) = self.apply_dual(self.kernel[self.svs,self.svs])
            self.threshold = 0.0

            idx = 0
            unl = pos = neg = neg2 = 0.0
            flag = flag_p = flag_n = False
            for i in self.svs:
                if (self.y[0,i]==0 and flag==False):
                    unl=thres[idx,0]
                    flag=True
                if (self.y[0,i]==0 and thres[idx,0]>unl): unl=thres[idx,0]
                if (self.y[0,i]>=1 and flag_p==False):
                    pos=thres[idx,0]
                    flag_p=True
                if (self.y[0,i]>=1 and thres[idx,0]>pos): pos=thres[idx,0]
                if (self.y[0,i]<=-1 and flag_n==False):
                    neg = neg2 =thres[idx,0]
                    flag_n=True
                if (self.y[0,i]<=-1 and thres[idx,0]>neg): neg=thres[idx,0]
                if (self.y[0,i]<=-1 and thres[idx,0]<neg2): neg2=thres[idx,0]
                idx += 1

            # now, check the possibilities
            if (flag==True):
                print('Threshold: unlabeled')
                self.threshold=unl
            if (flag_p==True and self.threshold<pos):
                print('Threshold: positive {0}'.format(pos))
                self.threshold=pos
            if (flag_n==True and self.threshold<neg):
                print('Threshold: negative {0}'.format(neg))
                self.threshold=neg
            if (flag_n == flag_p == True):
                self.threshold = 0.5*(pos + neg2)
                print('Threshold: middle {0}'.format(self.threshold))

        self.threshold = matrix(self.threshold)
        print('New threshold is {0}'.format(self.threshold))
        return SSAD.MSG_OK


    def get_threshold(self):
        return self.threshold


    def get_support_dual(self):
        return self.svs


    def get_alphas(self):
        return self.alphas


    def apply_dual(self, kernel):
        """ Application of dual trained ssad.
            kernel = Kernel.get_kernel(Y, X[:,cssad.svs], kernel_type, kernel_param)
        """
        # number of support vectors
        N = len(self.svs)

        # check number and dims of test data
        (tN,talphas) = kernel.size
        if (tN<1):
            print('Invalid test data')
            return 0, SSAD.MSG_ERROR

        # apply trained classifier
        prod = matrix([self.alphas[i,0]*self.cy[0,i] for i in self.svs],(N,1))
        res = matrix([dotu(kernel[i,:],prod) for i in range(tN)])
        return res, SSAD.MSG_OK
