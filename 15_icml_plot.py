import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import scipy.io as io

def plot_icml_pgm_results():
    files = [2.5,5,10,15,20,30]

    # annotation results
    rnames = ['SSVM (FS)','SSVM (Full)','HMAD (FS)','HMAD (Full)']
    rscores = np.zeros((len(rnames),len(files)))
    rvars = np.zeros((len(rnames),len(files)))

    # anomaly detection results
    anames = ['SSVM (FS)','SSVM (Full)','OcSvm Spectrum (1)','OcSvm Spectrum (2)','OcSvm Spectrum (3)', \
        'OcSvm Spectrum (FS)','HMAD (FS)','HMAD (Full)']
    ascores = np.zeros((len(anames),len(files)))
    avars = np.zeros((len(anames),len(files)))


    for f in range(len(files)):
        data = io.loadmat('icml_pgm_d{0}'.format(str(files[f])))
        res = data['res']
        auc = data['auc']

        for idx in range(len(rnames)):
            m = np.mean(res[rnames[idx]][0][0][:,0])
            v = np.std(res[rnames[idx]][0][0][:,0])
            rscores[idx,f] = m
            rvars[idx,f] = v
            print (rnames[idx], m, v)

        for idx in range(len(anames)):
            m = np.mean(auc[anames[idx]][0][0])
            v = np.std(auc[anames[idx]][0][0])
            ascores[idx,f] = m
            avars[idx,f] = v
            print (anames[idx], m, v)

    plt.figure(1)
    cnt = 0
    names = []
    style = ['-','-','-','-','-','--','--']
    marker = ['D','^','p','o','D','o','s']
    colors = [[0.,0.5,0.],[0.,0.75,0.],[0.,1.,0.],'y','m','b','k']
    for idx in [2,3,4,5,6,7]:
        names.append(anames[idx])
        if not idx==7 and not idx==6:
            plt.errorbar(files, ascores[idx,:], yerr=avars[idx,:], \
                fmt=style[cnt], color=colors[cnt], linewidth=4, alpha=0.7-(cnt*0.1), marker=marker[cnt], markersize=8)
        else:
            plt.errorbar(files, ascores[idx,:], yerr=avars[idx,:], \
                fmt=style[cnt], color=colors[cnt], linewidth=idx-2, alpha=1.0, marker=marker[cnt], markersize=8)

        plt.xticks(files, ['2.5%','5%','10%','15%','20%','30%'], fontsize=16)
        plt.yticks([0.58,0.6,0.7,0.8,0.9,1.0,1.02], ['','0.6','0.7','0.8','0.9','1.0',''], fontsize=16)
        cnt += 1
    plt.ylabel('AUC',fontsize=20)
    plt.xlabel('Outliers',fontsize=20)
    names[-2] = 'Hidden Markov Anomaly Detection (FS)'
    names[-1] = 'Hidden Markov Anomaly Detection'
    plt.legend(names,loc=3)
    plt.show()

    print '.....................................................'
    print np.array(rscores[[1,3],:]).T
    print '.....................................................'
    print np.array(rvars[[1,3],:]).T
    print '.....................................................'
    print ascores[[2,3,4,7],:]
    print '.....................................................'
    print('finished')


def plot_icml_wind_results():
    files = [2.5,5.0,10.0,15.0,20.0,30.0]

    # annotation results
    rnames = ['SSVM','HMAD']
    rnames = ['HMAD']
    rscores = np.zeros((len(rnames),len(files)))
    rvars = np.zeros((len(rnames),len(files)))

    # anomaly detection results
    anames = ['SSVM','OcSvm (Hist 2)','OcSvm (Hist 4)','OcSvm (Hist 8)','HMAD']
    anames = ['OcSvm (Hist 2)','OcSvm (Hist 4)','OcSvm (Hist 8)','HMAD']
    ascores = np.zeros((len(anames),len(files)))
    avars = np.zeros((len(anames),len(files)))


    for f in range(len(files)):
        data = io.loadmat('15_icml_wind_b{0}'.format(str(files[f])))
        res = data['res']
        auc = data['auc']

        for idx in range(len(rnames)):
            m = np.mean(res[rnames[idx]][0][0][:,0])
            v = np.std(res[rnames[idx]][0][0][:,0])
            rscores[idx,f] = m
            rvars[idx,f] = v
            print (rnames[idx], m, v)

        for idx in range(len(anames)):
            m = np.mean(auc[anames[idx]][0][0])
            v = np.std(auc[anames[idx]][0][0])
            ascores[idx,f] = m
            avars[idx,f] = v
            print (anames[idx], m, v)

    plt.figure(1)
    cnt = 0
    names = []
    style = ['-','-','-','--','D','o','s']
    marker = ['D','^','p','o','D','o','s']
    colors = [[0.,0.5,0.],[0.,0.75,0.],[0.,1.,0.],'b','m','y','k']
    for idx in [0,1,2,3]:
        names.append(anames[idx])
        if not idx==7:
            plt.errorbar(files, ascores[idx,:], yerr=avars[idx,:], \
                fmt=style[cnt], color=colors[cnt], elinewidth=1, linewidth=4, alpha=0.7-(cnt*0.1), marker=marker[cnt], markersize=8)
        else:
            plt.errorbar(files, ascores[idx,:], yerr=avars[idx,:], \
                fmt=style[cnt], color=colors[cnt], elinewidth=1, linewidth=4, alpha=1.0, marker=marker[cnt], markersize=8)

        plt.xticks(files, ['2.5%','5%','10%','15%','20%','30%'], fontsize=16)
        plt.yticks([0.58,0.6,0.7,0.8,0.9,1.0,1.02], ['','0.6','0.7','0.8','0.9','1.0',''], fontsize=16)
        cnt += 1
    plt.ylabel('AUC',fontsize=20)
    plt.xlabel('Outliers',fontsize=20)

    names = ['OcSvm (Hist 2)','OcSvm (Hist 4)','OcSvm (Hist 8)','Hidden Markov Anomaly Detection']
    plt.legend(names,loc=4)
    plt.show()
    print('finished')



def plot_toy_results():
    files = ['15_icml_toy_ad_03','15_icml_toy_anno_01']

    # annotation results
    rnames = ['SSVM','HMAD']
    rscores = np.zeros((len(rnames),len(files)))
    rvars = np.zeros((len(rnames),len(files)))

    # anomaly detection results
    anames = ['Bayes Classifier','HMAD','OcSvm linear','OcSvm linear (normalized)', \
        'OcSvm RBF $\sigma^2=0.1$','OcSvm RBF $\sigma^2=1.0$','OcSvm RBF $\sigma^2=2.0$', \
        'OcSvm RBF $\sigma^2=4.0$','OcSvm RBF $\sigma^2=10.0$']
    #anames = ['Bayes Classifier','HMAD','OcSvm linear','OcSvm linear (normalized)', \
    #    'OcSvm RBF $\sigma^2=0.1$','OcSvm RBF $\sigma^2=1.0$','OcSvm RBF $\sigma^2=2.0$', \
    #    'OcSvm RBF $\sigma^2=4.0$']
    ascores = np.zeros((len(anames),len(files)))
    avars = np.zeros((len(anames),len(files)))

    auc = io.loadmat('{0}'.format(files[0]))
    res = io.loadmat('{0}'.format(files[1]))

    print res.keys()
    blocks = 101-auc['BLOCKS'][0]
    lens = len(auc['BLOCKS'][0])
    reps = float(auc['REPS'][0][0])

    plt.figure(1)

    plt.subplot(1,2,2)
    style = ['-','--','-','-']
    marker = ['D','o','^','o']
    colors = ['r','b','m','c']

    #blocks = range(len(blocks))
    cnt = 0
    plt.errorbar(blocks, auc['mbayes_auc'][0], yerr=np.sqrt(auc['vbayes_auc'][0]), \
        fmt=style[cnt], color=colors[cnt], linewidth=4, alpha=1.0, marker=marker[cnt], markersize=10)
    cnt = 1
    plt.errorbar(blocks, auc['mauc'][0], yerr=(auc['vauc'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=6, alpha=0.6, marker=marker[cnt], markersize=10,
        elinewidth=2)
    cnt = 2
    plt.errorbar(blocks, auc['mbase_auc'][0], yerr=(auc['vbase_auc'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=3, alpha=0.8, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 2
    plt.errorbar(blocks, auc['mbase_auc1'][0], yerr=(auc['vbase_auc1'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=5, alpha=0.4, marker=marker[cnt], markersize=10, \
        elinewidth=2)


    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc2'][0], yerr=(auc['vbase_auc2'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=2, alpha=1.0, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc3'][0], yerr=(auc['vbase_auc3'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=3, alpha=0.8, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc4'][0], yerr=(auc['vbase_auc4'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=4, alpha=0.6, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc5'][0], yerr=(auc['vbase_auc5'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=5, alpha=0.4, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc6'][0], yerr=(auc['vbase_auc6'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=6, alpha=0.2, marker=marker[cnt], markersize=10, \
        elinewidth=2)


    plt.title('Anomaly Detection', fontsize=28)
    plt.xscale('linear')
    print blocks
    plt.xticks(blocks, ['0','20','40','60','80','','','','100'][::-1], fontsize=18)
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.05], ['','0.2','0.4','0.6','0.8','1.0',''], fontsize=18)
    plt.xlim((1.,100.))

    plt.ylabel('AUC',fontsize=24)
    plt.xlabel('Structure in %',fontsize=24)
    #plt.legend(anames,loc=2, fontsize=18)


    plt.subplot(1,2,1)
    cnt = 0
    plt.errorbar(blocks, res['conts_base'][:,0], yerr=(res['var_base'][:,0]), elinewidth=2,\
        fmt=style[cnt], color=colors[cnt], linewidth=4, alpha=1.0, marker=marker[cnt], markersize=8)
    cnt = 1
    plt.errorbar(blocks, res['conts'][:,0], yerr=(res['var'][:,0]), elinewidth=2,\
        fmt=style[cnt], color=colors[cnt], linewidth=6, alpha=0.6, marker=marker[cnt], markersize=8)
    plt.title('Annotation', fontsize=28)
    #plt.xscale('log')
    plt.xticks(blocks, ['0','20','40','60','80','','','','100'][::-1], fontsize=18)
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.05], ['0.0','0.2','0.4','0.6','0.8','1.0',''], fontsize=18)
    plt.ylabel('F-score',fontsize=24)
    plt.xlabel('Structure in %',fontsize=24)    
    plt.legend(rnames, loc=4, fontsize=18)

    plt.show()



    print auc['mbayes_auc']
    print auc['mauc']
    print auc['mbase_auc']



def plot_toy_ad_results():
    files = ['15_icml_toy_ad_05']
    # anomaly detection results
    anames = ['Bayes Classifier','HMAD','OcSvm linear','OcSvm linear (normalized)', \
        'OcSvm RBF $\sigma^2=0.1$','OcSvm RBF $\sigma^2=1.0$','OcSvm RBF $\sigma^2=2.0$', \
        'OcSvm RBF $\sigma^2=4.0$','OcSvm RBF $\sigma^2=10.0$', 'OcSvm 2-Hist', 'OcSvm 4-Hist']
    anames = ['Bayes Classifier','HMAD','OcSvm linear','OcSvm linear (normalized)', \
        'OcSvm RBF $\sigma^2=0.1$','OcSvm RBF $\sigma^2=1.0$','OcSvm RBF $\sigma^2=2.0$', \
        'OcSvm RBF $\sigma^2=4.0$', 'OcSvm 2-Hist', 'OcSvm 4-Hist']
    ascores = np.zeros((len(anames),len(files)))
    avars = np.zeros((len(anames),len(files)))

    auc = io.loadmat('{0}'.format(files[0]))

    blocks = auc['BLOCKS'][0][::-1]
    blocks = auc['BLOCKS'][0]
    print auc
    print blocks
    lens = len(auc['BLOCKS'][0])
    reps = float(auc['REPS'][0][0])
    print reps

    plt.figure(1)
    style = ['-','--','-','-','-']
    marker = ['D','o','^','o','s']
    colors = ['r','b','m','c','g']

    cnt = 0
    plt.errorbar(blocks, auc['mbayes_auc'][0], yerr=np.sqrt(auc['vbayes_auc'][0]), \
        fmt=style[cnt], color=colors[cnt], linewidth=4, alpha=1.0, marker=marker[cnt], markersize=10)
    cnt = 1
    plt.errorbar(blocks, auc['mauc'][0], yerr=(auc['vauc'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=6, alpha=0.6, marker=marker[cnt], markersize=10,
        elinewidth=2)
    cnt = 2
    plt.errorbar(blocks, auc['mbase_auc'][0], yerr=(auc['vbase_auc'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=3, alpha=0.8, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 2
    plt.errorbar(blocks, auc['mbase_auc1'][0], yerr=(auc['vbase_auc1'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=5, alpha=0.4, marker=marker[cnt], markersize=10, \
        elinewidth=2)


    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc2'][0], yerr=(auc['vbase_auc2'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=2, alpha=1.0, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc3'][0], yerr=(auc['vbase_auc3'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=3, alpha=0.8, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc4'][0], yerr=(auc['vbase_auc4'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=4, alpha=0.6, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 3
    plt.errorbar(blocks, auc['mbase_auc5'][0], yerr=(auc['vbase_auc5'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=5, alpha=0.4, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    #cnt = 3
    #plt.errorbar(blocks, auc['mbase_auc6'][0], yerr=(auc['vbase_auc6'][0]*reps/(reps-1.0)), \
    #    fmt=style[cnt], color=colors[cnt], linewidth=6, alpha=0.2, marker=marker[cnt], markersize=10, \
    #    elinewidth=2)

    cnt = 4
    plt.errorbar(blocks, auc['mbase_hist_auc1'][0], yerr=(auc['vbase_hist_auc1'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=5, alpha=0.4, marker=marker[cnt], markersize=10, \
        elinewidth=2)
    cnt = 4
    plt.errorbar(blocks, auc['mbase_hist_auc2'][0], yerr=(auc['vbase_hist_auc2'][0]*reps/(reps-1.0)), \
        fmt=style[cnt], color=colors[cnt], linewidth=6, alpha=0.2, marker=marker[cnt], markersize=10, \
        elinewidth=2)

    plt.title('Anomaly Detection', fontsize=28)
    plt.xscale('linear')
    #plt.xticks(blocks, ['1','2','5','10','20','40','60','80','100'][::-1], fontsize=18)
    plt.xticks(blocks, ['1','2','5','10','20','40','60','80','100'], fontsize=18)
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.02], ['','0.2','0.4','0.6','0.8','1.0',''], fontsize=18)
    #plt.xlim((1.,100.))

    plt.ylabel('AUC',fontsize=24)
    plt.xlabel('Structure in %',fontsize=24)
    plt.legend(anames,loc=3, fontsize=18)
    plt.show()


def plot_icml_toy_ad_results():
    files = ['15_icml_toy_ad_b0']
    # anomaly detection results
    data = io.loadmat('{0}'.format(files[0]))

    blocks = data['BLOCKS'][0][::-1]
    blocks = data['BLOCKS'][0]
    #blocks = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
    print blocks
    lens = len(data['BLOCKS'][0])
    reps = float(data['REPS'][0][0])
    print reps
    names = data['names']
    print names
    aucs = data['aucs']
    stds = data['stds']
    varis = data['varis']
    print aucs

    plt.figure(1)
    style = ['-','--','-','.-','--','-','-','-','-','--']
    marker = ['D','o','^','D','s','^','D','s','s','o']
    colors = ['r','b','m','m','m','c','c','c','g','g']
    alphas = [1. ,1. ,.8 ,.6 ,.4 ,.8 ,.6 ,.4 ,1. ,.6 ]
    widths = [2  ,4  ,1  ,4  ,8  ,1  ,4  ,8  ,1  ,8  ]

    for i in range(len(names)):
        plt.errorbar(blocks, aucs[i,:], yerr=stds[i,:], \
            fmt=style[i], color=colors[i], linewidth=widths[i], alpha=alphas[i], marker=marker[i], markersize=10)

    plt.title('Anomaly Detection', fontsize=28)
    plt.xscale('log')
    #plt.xticks(blocks, ['1','2','5','10','20','40','60','80','100'][::-1], fontsize=18)
    #plt.xticks(blocks, ['1','2','5','10','20','40','60','80','100'], fontsize=18)
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.02], ['','0.2','0.4','0.6','0.8','1.0',''], fontsize=18)
    #plt.xlim((1.,100.))

    plt.ylabel('AUC',fontsize=24)
    plt.xlabel('Structure in %',fontsize=24)
    plt.legend(names,loc=3, fontsize=18)
    plt.show()


if __name__ == '__main__':
    plot_icml_toy_ad_results()