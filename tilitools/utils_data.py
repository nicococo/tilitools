import numpy as np
import cvxopt as co


def load_mnist_dataset():
    import torchvision.datasets as datasets

    mnist_train = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=None)
    test_labels = np.array([mnist_test[i][1].numpy() for i in range(len(mnist_test))], dtype=np.int)
    train_labels = np.array([mnist_train[i][1].numpy() for i in range(len(mnist_train))], dtype=np.int)
    test = np.array([np.asarray(mnist_test[i][0]).reshape(28*28) for i in range(len(mnist_test))], dtype=np.float)
    train = np.array([np.asarray(mnist_train[i][0]).reshape(28*28) for i in range(len(mnist_train))], dtype=np.float)
    train /= 255.  # normalize data to be in range [0,1]
    test /= 255.
    return train, train_labels, test, test_labels, [28, 28]


def load_fashion_mnist_dataset():
    import torchvision.datasets as datasets

    mnist_train = datasets.FashionMNIST(root='./data/fashion-mnist', train=True, download=True, transform=None)
    mnist_test = datasets.FashionMNIST(root='./data/fashion-mnist', train=False, download=True, transform=None)
    test_labels = np.array([mnist_test[i][1].numpy() for i in range(len(mnist_test))], dtype=np.int)
    train_labels = np.array([mnist_train[i][1].numpy() for i in range(len(mnist_train))], dtype=np.int)
    test = np.array([np.asarray(mnist_test[i][0]).reshape(28*28) for i in range(len(mnist_test))], dtype=np.float)
    train = np.array([np.asarray(mnist_train[i][0]).reshape(28*28) for i in range(len(mnist_train))], dtype=np.float)
    train /= 255.  # normalize data to be in range [0,1]
    test /= 255.
    return train, train_labels, test, test_labels, [28, 28]


def load_emnist_dataset():
    import torchvision.datasets as datasets

    mnist_train = datasets.EMNIST(root='./data/emnist', split='balanced', train=True, download=True, transform=None)
    mnist_test = datasets.EMNIST(root='./data/emnist', split='balanced', train=False, download=True, transform=None)
    test_labels = np.array([mnist_test[i][1].numpy() for i in range(len(mnist_test))], dtype=np.int)
    train_labels = np.array([mnist_train[i][1].numpy() for i in range(len(mnist_train))], dtype=np.int)
    test = np.array([np.asarray(mnist_test[i][0]).reshape(28*28) for i in range(len(mnist_test))], dtype=np.float)
    train = np.array([np.asarray(mnist_train[i][0]).reshape(28*28) for i in range(len(mnist_train))], dtype=np.float)
    train /= 255.  # normalize data to be in range [0,1]
    test /= 255.
    return train, train_labels, test, test_labels, [28, 28]


def load_cifar10_dataset():
    import torchvision.datasets as datasets

    cifar_train = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=None)
    cifar_test = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=None)
    test_labels = np.array([cifar_test[i][1] for i in range(len(cifar_test))], dtype=np.int)
    train_labels = np.array([cifar_train[i][1] for i in range(len(cifar_train))], dtype=np.int)
    test = np.array([np.asarray(cifar_test[i][0].convert('F')).reshape(32*32) for i in range(len(cifar_test))], dtype=np.float)
    train = np.array([np.asarray(cifar_train[i][0].convert('F')).reshape(32*32) for i in range(len(cifar_train))], dtype=np.float)
    train /= 255.  # normalize data to be in range [0,1]
    test /= 255.
    return train, train_labels, test, test_labels, [32, 32]


def get_gaussian(num, dims=2, means=[0,0], vars=[1,1]):
    data = np.random.multivariate_normal(means, np.eye(dims), num)
    return data


def get_2state_gaussian_seq(lens,dims=2,means1=[2,2,2,2],means2=[5,5,5,5],vars1=[1,1,1,1],vars2=[1,1,1,1],anom_prob=1.0):
    seqs = np.zeros((dims, lens))
    lbls = np.zeros((1, lens), dtype=np.int)
    marker = 0

    # generate first state sequence
    for d in range(dims):
        seqs[d, :] = np.random.randn(lens)*vars1[d] + means1[d]

    prob = np.random.uniform()
    if prob < anom_prob:
        # add second state blocks
        while True:
            max_block_len = 0.6*lens
            min_block_len = 0.1*lens
            block_len = np.int(max_block_len*np.random.uniform()+3)
            block_start = np.int(lens*np.random.uniform())
            if block_len - (block_start+block_len-lens)-3 > min_block_len:
                break

        block_len = min( [block_len, block_len - (block_start+block_len-lens)-3] )
        lbls[block_start:block_start+block_len-1] = 1
        marker = 1
        for d in range(dims):
            seqs[d,block_start:block_start+block_len-1] = np.random.randn(1,block_len-1)*vars2[d] + means2[d]
    return seqs, lbls, marker



def get_2state_anom_seq(lens, comb_block_len, anom_prob=1.0, num_blocks=1):
    seqs = co.matrix(0.0, (1, lens))
    lbls = co.matrix(0, (1, lens))
    marker = 0

    # generate first state sequence, gaussian noise 0=mean, 1=variance
    seqs = np.zeros((1, lens))
    lbls = np.zeros((1, lens))
    bak = seqs.copy()

    prob = np.random.uniform()
    if prob < anom_prob:

        # add second state blocks
        block_len = np.int(np.floor(comb_block_len / float(num_blocks)))
        marker = 1

        # add a single block
        blen = 0
        for b in range(np.int(num_blocks)):
            if (b==num_blocks-1 and b>1):
                block_len = np.round(comb_block_len-blen)

            isDone = False
            while isDone == False:
                start = np.int(np.random.uniform()*float(lens-block_len+1))
                if np.sum(lbls[0,start:start+block_len]) == 0:
                    lbls[0, start:start+block_len] = 1
                    seqs[0, start:start+block_len] = bak[0, start:start+block_len]+4.0
                    isDone = True
                    break
            blen += block_len
        # print('Anomamly block lengths (target/reality)= {0}/{1} '.format(comb_block_len, blen))
    return seqs, lbls, marker
