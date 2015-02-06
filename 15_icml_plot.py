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
    style = ['-','-','-','-','-','--','--','--']
    marker = ['D','^','p','o','D','o','s','s']
    colors = [[0.,0.5,0.],[0.,0.75,0.],[0.,1.,0.],'g','m','b','k']
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
    plt.ylabel('Detection accuracy [in AUC]',fontsize=18)
    plt.xlabel('Percentage of anomalous data',fontsize=18)
    #names[-2] = 'Hidden Markov Anomaly Detection (FS)'
    #names[-1] = 'Hidden Markov Anomaly Detection'
    names[-1] = 'HMAD'
    legnames = []
    for name in names:
        if 'OcSvm' in name:
            name = 'OC-SVM' + name[5:]
            print name
        legnames.append(name)
    plt.legend(legnames,loc=3,fontsize=18,framealpha=0.7,fancybox=True)
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
    anames = ['SSVM','OcSvm (Hist 4)','OcSvm (Hist 8)','OcSvm (Hist 12)','HMAD']
    anames = ['OcSvm (Hist 4)','OcSvm (Hist 8)','OcSvm (Hist 16)','HMAD']
    ascores = np.zeros((len(anames),len(files)))
    avars = np.zeros((len(anames),len(files)))

    for f in range(len(files)):
        data = io.loadmat('15_icml_wind_c{0}'.format(str(files[f])))
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
    colors = [[0.1,0.4,0.6],[0.1,0.5,0.8],[0.2,0.8,1.0],'b','m','y','k']
    for idx in [0,1,2,3]:
        names.append(anames[idx])
        if not idx==3:
            plt.errorbar(files, ascores[idx,:], yerr=avars[idx,:], \
                fmt=style[cnt], color=colors[cnt], elinewidth=1, linewidth=2+idx*2, alpha=0.7-(cnt*0.2), marker=marker[cnt], markersize=8)
        else:
            plt.errorbar(files, ascores[idx,:], yerr=avars[idx,:], \
                fmt=style[cnt], color=colors[cnt], elinewidth=1, linewidth=5, alpha=0.9, marker=marker[cnt], markersize=8)

        plt.xticks(files, ['2.5%','5%','10%','15%','20%','30%'], fontsize=16)
        plt.yticks([0.58,0.6,0.7,0.8,0.9,1.0,1.02], ['','0.6','0.7','0.8','0.9','1.0',''], fontsize=16)
        cnt += 1
    plt.ylabel('Detection accuracy [in AUC]',fontsize=18)
    plt.xlabel('Percentage of anomalous data',fontsize=18)

    names[:3] = ['OC-SVM (Hist 4)','OC-SVM (Hist 8)','OC-SVM (Hist 16)']
    #names[-1] = 'Hidden Markov Anomaly Detection'
    plt.legend(names,loc=4, fontsize=18, fancybox=True, framealpha=0.7)
    plt.show()
    print('finished')




def plot_icml_toy_results():
    filename = '15_icml_toy_runtime_b0'
    filename = '15_icml_toy_ad_b0'
    filename = '15_icml_toy_adfrac_b0'

    # anomaly detection results
    data = io.loadmat('{0}'.format(filename))

    blocks = data['BLOCKS'][0][::-1]
    blocks = data['BLOCKS'][0] # for ad experiment
    if 'adfrac' in filename:
        blocks = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3] # for adfrac experiment
    print blocks
    lens = len(data['BLOCKS'][0])
    reps = float(data['REPS'][0][0])
    print reps
    names = data['names']
    print names
    if 'runtime' in filename:
        aucs = data['times']
    else:
        aucs = data['aucs']
    stds = data['stds']
    varis = data['varis']
    print aucs

    plt.figure(1)
    style = ['-','--','-','.-','--','-','-','-','-','--']
    marker = ['D','o','^','D','s','^','D','s','s','o']
    colors = ['r','b','m','m','m','c','c','c','g','g']
    alphas = [1. ,0.9,.8 ,.6 ,.4 ,.8 ,.6 ,.4 ,1. ,.6 ]
    widths = [2  ,4  ,1  ,4  ,8  ,1  ,4  ,8  ,1  ,8  ]

    remain_idx = [0,1,3,6,8]
    for i in range(len(names)):
        if not i in remain_idx:
            continue

        if i<len(names)-2:
            names[i] = names[i][:len(names[i])-5]
        plt.errorbar(blocks, aucs[i,:], yerr=stds[i,:], \
            fmt=style[i], color=colors[i], linewidth=widths[i], alpha=alphas[i], marker=marker[i], markersize=10)

    #plt.title('Anomaly Detection', fontsize=28)
    if  '_ad_' in filename:
        plt.xscale('log')
        plt.xticks(blocks, ['0%','2%','5%','10%','20%','40%','60%','100%'], fontsize=18)
        plt.xlabel('Percentage of disorganization',fontsize=22)
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.02], ['','0.2','0.4','0.6','0.8','1.0',''], fontsize=18)
        plt.ylabel('Detection accuracy [in AUC]',fontsize=22)
    
    if 'adfrac' in filename:
        plt.xticks(blocks, ['2.5%','5%','10%','15%','20%','30%'], fontsize=18)
        plt.xlim((0.025,0.3))
        plt.xlabel('Percentage of anomalous data',fontsize=22)
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.02], ['','0.2','0.4','0.6','0.8','1.0',''], fontsize=18)
        plt.ylabel('Detection accuracy [in AUC]',fontsize=22)

    if  'runtime' in filename:
        plt.yscale('log')
        plt.xticks(blocks, fontsize=18)
        plt.xlabel('Number of training examples',fontsize=22)
        plt.yticks([0.00001,0.001,0.01,1.0,1000.], fontsize=18)
        plt.ylabel('Time in [sec]',fontsize=22)
        names[1] = 'HMAD'
        names[4] += ')'
        names = names[remain_idx]
        legnames = []
        for name in names:
            if 'OcSvm' in name:
                name = 'OC-SVM' + name[5:]
                if 'Linear' in name:
                   name = name[:len(name)-6]
                print name
            legnames.append(name)
        plt.legend(legnames,loc=4, fancybox=True, framealpha=0.7, fontsize=20)

    plt.show()



def plot_icml_toy_seqs():
    from toydata import ToyData
    lens = 600
    blocks = [1,2,5,10,20,40,60,100]
    data = []
    lbl = []
    for i in blocks:
        (exm, label, marker) = ToyData.get_2state_anom_seq(lens, 120, anom_prob=1.0, num_blocks=i)
        data.append(exm)
        lbl.append(label)

    plt.figure(1)
    style = ['-','--','-','.-','--','-','-','-','-','--']
    marker = ['D','o','^','D','s','^','D','s','s','o']
    colors = ['r','b','m','m','m','c','c','c','g','g']
    alphas = [1. ,0.9,.8 ,.6 ,.4 ,.8 ,.6 ,.4 ,1. ,.6 ]
    widths = [2  ,4  ,1  ,4  ,8  ,1  ,4  ,8  ,1  ,8  ]

    ys = []
    for i in [0,-1]:
        ys.append(i*10)
        plt.plot(range(lens), np.array(data[i]).T + ys[-1], \
            color='r', linewidth=2, alpha=0.6, marker='')
        plt.plot(range(lens), np.array(lbl[i]).T + ys[-1], \
            color='k', linewidth=2, alpha=0.8, marker='')

    #plt.yscale('log')
    plt.xticks([0,300,600], ['0','300','600'], fontsize=16)
    plt.yticks(ys, ['0%','100%'], fontsize=16)
    plt.ylabel('Percentage of disorganization',fontsize=18)
    plt.xlabel('Sequence position',fontsize=18)
    plt.legend(['Noisy observations','True state sequence'], fontsize=18, fancybox=True, framealpha=0.7)
    plt.show()



if __name__ == '__main__':
    plot_icml_pgm_results()