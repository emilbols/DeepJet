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

listbranch = ['isB','isC','isBB','isGBB','isLeptonicB','isLeptonicB_C','isUD','isS','isGCC','isCC']
listbranch1 = ['prob_isB','prob_isBB']


def draw_roc_mean(df, df2, label, color, draw_unc=False, ls='-', draw_auc=True):
    newx = np.logspace(-3, 0, 50)
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

def draw_roc(df, df2, label, color, draw_unc=False, ls='-', draw_auc=True):
    newx = np.logspace(-3, 0, 50)
    tprs = pd.DataFrame()
    scores = []
    tmp_fpr, tmp_tpr, _ = roc_curve(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isBB']+df2['prob_isB'])
    scores.append(
        roc_auc_score(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isB']+df2['prob_isBB'])
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

def spit_out_roc(df, df2, label, color, draw_unc=False, ls='-', draw_auc=True):
    newx = np.logspace(-3, 0, 50)
    tprs = pd.DataFrame()
    scores = []
    tmp_fpr, tmp_tpr, _ = roc_curve(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isBB']+df2['prob_isB'])
    scores.append(
        roc_auc_score(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isB']+df2['prob_isBB'])
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

truthfile = open('/data/ml/ebols/PredictionTest1/tree_association.txt','r')
rfile2 = ROOT.TChain("deepntuplizer/tree")
for line in truthfile:
    if len(line) < 1: continue
    file1name=str(line.split(' ')[0])
    rfile2.Add(file1name)

truth = root_numpy.tree2array(rfile2, branches = listbranch)

for z in range(1,13):
            infile = open('/data/ml/ebols/PredictionTest' + str(z) +'/tree_association.txt','r')
            rfile1 = ROOT.TChain("tree")
            for line in infile:
                        if len(line) < 1: continue
                        file2name=str(line.split(' ')[1].split('.')[0])
                        file2name = file2name + '.root'
                        rfile1.Add(file2name)
            pred.append(root_numpy.tree2array(rfile1, branches = listbranch1))


meantag = np.mean( np.array([ pred[z-1]['prob_isBB']+pred[z-1]['prob_isB'] for z in range(1,13) ]), axis=0 )
plt.figure(figsize=(18,10))
tprs = []
newx = []
for z in range(1,13):
    temp1, temp2 = spit_out_roc(truth, pred[z-1], str(z), 'blue')
    tprs.append(temp1)
    newx.append(temp2)
meantprs = np.mean( np.array([ tprs[z-1] for z in range(1,13) ]), axis=0 )
plt.plot(meantprs, newx[0], label='mean of rocs', c='blue', ls='-')

draw_roc_mean(truth, meantag, 'roc of means', 'green',draw_auc=False)
plt.grid()
plt.yscale("log")
plt.legend()
plt.savefig("combine.png")
