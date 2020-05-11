import os
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import root_numpy
import ROOT

listbranch = ['isB','isC','isBB','isGBB','isLeptonicB','isLeptonicB_C','isUD','isS','isGCC','isCC','isG','jet_pt']
listbranch1 = ['prob_isB','prob_isBB','prob_isLeptB']


def draw_roc_mean(df, df2, label, color, draw_unc=False, ls='-', draw_auc=True):
    newx = np.logspace(-3, 0, 100)
    tprs = pd.DataFrame()
    scores = []
    tmp_fpr, tmp_tpr, _ = roc_curve(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2)
    scores.append(
        roc_auc_score(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2)
    )
    coords = pd.DataFrame()
    coords['fpr'] = tmp_fpr
    coords['tpr'] = tmp_tpr
    clean = coords.drop_duplicates(subset=['fpr'])
    spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
    tprs = spline(newx)
    scores = np.array(scores)
    auc = ' AUC: %.3f +/- %.3f' % (scores.mean(), scores.std()) if draw_auc else ''
        
    plt.plot(tprs, newx, label=label + auc, c=color, ls=ls)

def draw_roc(df, df2, label, color, draw_unc=False, ls='-', draw_auc=True, flavour = False):
    newx = np.logspace(-3, 0, 100)
    tprs = pd.DataFrame()
    scores = []
    if flavour:
        cs = ( (df['isC'] == 0) & (df['isCC'] == 0) & (df['isGCC'] == 0) & (df['jet_pt'] > 30) )
    else:
        cs = ( (df['isUD'] == 0) & (df['isS'] == 0) & (df['isG'] == 0) & (df['jet_pt'] > 30) )
    df = df[cs]
    df2 = df2[cs]
    tmp_fpr, tmp_tpr, _ = roc_curve(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isBB']+df2['prob_isB']+df2['prob_isLeptB'])
    scores.append(
        roc_auc_score(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isB']+df2['prob_isBB']+df2['prob_isLeptB'])
    )
    coords = pd.DataFrame()
    coords['fpr'] = tmp_fpr
    coords['tpr'] = tmp_tpr
    clean = coords.drop_duplicates(subset=['fpr'])
    spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
    tprs = spline(newx)
    scores = np.array(scores)
    auc = ' AUC: %.3f +/- %.3f' % (scores.mean(), scores.std()) if draw_auc else ''
        
    plt.plot(tprs, newx, label=label + auc, c=color, ls=ls)

def spit_out_roc(df, df2, label, color, draw_unc=False, ls='-', draw_auc=True, flavour = False):
    newx = np.logspace(-3, 0, 100)
    tprs = pd.DataFrame()
    scores = []
    if flavour:
        cs = ( (df['isC'] == 0) & (df['isCC'] == 0) & (df['isGCC'] == 0) & (df['jet_pt'] > 30) )
    else:
        cs = ( (df['isUD'] == 0) & (df['isS'] == 0) & (df['isG'] == 0) & (df['jet_pt'] > 30) )
    df = df[cs]
    df2 = df2[cs]
    tmp_fpr, tmp_tpr, _ = roc_curve(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isBB']+df2['prob_isB']+df2['prob_isLeptB'])
    scores.append(
        roc_auc_score(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isB']+df2['prob_isBB']+df2['prob_isLeptB'])
    )
    coords = pd.DataFrame()
    coords['fpr'] = tmp_fpr
    coords['tpr'] = tmp_tpr
    clean = coords.drop_duplicates(subset=['fpr'])
    spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
    tprs = spline(newx)
    scores = np.array(scores)
    auc = ' AUC: %.3f +/- %.3f' % (scores.mean(), scores.std()) if draw_auc else ''
        
    return tprs, newx


pred = []

truthfile = open('/eos/user/e/ebols/Predictors/PredictionDeepFlavour110XTTBar/tree_association.txt','r')
rfile2 = ROOT.TChain("deepntuplizer/tree")
for line in truthfile:
    if len(line) < 1: continue
    file1name=str(line.split(' ')[0])
    rfile2.Add(file1name)

truth = root_numpy.tree2array(rfile2, branches = listbranch)

for z in range(1,5):
            infile = open('/eos/user/e/ebols/Predictors/PredictionDeepFlavour' + str(z) +'10XTTBar/tree_association.txt','r')
            rfile1 = ROOT.TChain("tree")
            for line in infile:
                        if len(line) < 1: continue
                        file2name=str(line.split(' ')[1].split('.')[0])
                        file2name = file2name + '.root'
                        rfile1.Add(file2name)
            pred.append(root_numpy.tree2array(rfile1, branches = listbranch1))


meantag = np.mean( np.array([ pred[z-1]['prob_isBB']+pred[z-1]['prob_isB']+pred[z-1]['prob_isLeptB'] for z in range(1,5) ]), axis=0 )
plt.figure(figsize=(18,10))
tprs = []
newx = []
tprs2 = []
newx2 = []

for z in range(1,5):
    #draw_roc(truth,pred[z-1],'','blue',draw_auc=False, flavour = True)
    #draw_roc(truth,pred[z-1],'','red',draw_auc=False, flavour = False)
    temp1, temp2 = spit_out_roc(truth, pred[z-1], str(z), 'blue', flavour = True)
    temp3, temp4 = spit_out_roc(truth, pred[z-1], str(z), 'red', flavour = False)
    tprs.append(temp1)
    newx.append(temp2)
    tprs2.append(temp3)
    newx2.append(temp4)

meantprs = np.mean( np.array([ tprs[z-1] for z in range(1,5) ]), axis=0 )
stdtprs = np.std( np.array([ tprs[z-1] for z in range(1,5) ]), axis=0, dtype=np.float64 )
meantprs2 = np.mean( np.array([ tprs2[z-1] for z in range(1,5) ]), axis=0 )
stdtprs2 = np.std( np.array([ tprs2[z-1] for z in range(1,5) ]), axis=0, dtype=np.float64 )

plt.plot(meantprs, newx[0], label='mean of rocs, b vs light', c='blue', ls='-')
plt.plot(meantprs2, newx2[0], label='mean of rocs, b vs c', c='red', ls='-')

#plt.fill_betweenx(newx2[0],meantprs2-2*stdtprs2,meantprs2+2*stdtprs2, facecolor='lightskyblue')
#plt.fill_betweenx(newx[0],meantprs-2*stdtprs,meantprs+2*stdtprs, facecolor='salmon')
plt.fill_betweenx(newx2[0],meantprs2-stdtprs2,meantprs2+stdtprs2)
plt.fill_betweenx(newx[0],meantprs-stdtprs,meantprs+stdtprs)

#draw_roc_mean(truth, meantag, 'roc of means, b vs all', 'red',draw_auc=False)
plt.grid(which='both')
plt.yscale("log")
plt.ylim([0.001,1.0])
plt.xlim([0.3,1.0])
plt.xlabel('b-jet efficiency',fontsize=20)
plt.ylabel('misid. probability',fontsize=20)
params = {'legend.fontsize': 32,
          'legend.handlelength': 4}
plt.rcParams.update(params)
plt.legend()
plt.savefig("DeepFlavour_Multitraining1Sig.png")
