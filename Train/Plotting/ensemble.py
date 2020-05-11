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
from ROOT import TCanvas, TGraph

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
        
    return tprs, newx, scores.mean()


pred = []

truthfile = open('/afs/cern.ch/work/e/ebols/private/Prediction_DFSkim_NoAss_10XTTBarHad/tree_association.txt','r')
rfile2 = ROOT.TChain("deepntuplizer/tree")
for line in truthfile:
    if len(line) < 1: continue
    file1name=str(line.split(' ')[0])
    if not any(x in file1name for x in ['97','98','99']):
        print file1name
        rfile2.Add(file1name)

truth = root_numpy.tree2array(rfile2, branches = listbranch)
filelist =['/afs/cern.ch/work/e/ebols/private/Prediction_DFSkim_NoAss_10XTTBarHad/tree_association.txt',
           '/afs/cern.ch/work/e/ebols/private/Predict_more_tracks_wide_E32/tree_association.txt',
           '/eos/user/e/ebols/Predictors/PredictionDeepFlavour110XTTBar/tree_association.txt',
           '/eos/user/e/ebols/Predictors/PredictionDeepFlavour210XTTBar/tree_association.txt',
           '/eos/user/e/ebols/Predictors/PredictionDeepFlavour310XTTBar/tree_association.txt',
           '/eos/user/e/ebols/Predictors/PredictionDeepFlavour410XTTBar/tree_association.txt',
           '/eos/user/e/ebols/DeepGraphAttempt/tree_association.txt',
           '/afs/cern.ch/work/e/ebols/private/Predict_more_tracks_E35/tree_association.txt',
           '/afs/cern.ch/work/e/ebols/private/DF_pruned_Predict10XTTBarHad/tree_association.txt',
           '/afs/cern.ch/work/e/ebols/private/DF2_pruned_Predict10XTTBarHad/tree_association.txt',
           '/eos/user/e/ebols/Prediction_FinalDeepFlavour_frac0p50_10XTTBar/tree_association.txt',
           '/eos/user/e/ebols/PermInvariantPrediction2/tree_association.txt'
]
coeff = np.array([])
for z in filelist:
            infile = open(z,'r')
            rfile1 = ROOT.TChain("tree")
            for line in infile:
                        if len(line) < 1: continue
                        file2name=str(line.split(' ')[1].split('.r')[0])
                        file2name = file2name + '.root'
                        if not any(x in file2name for x in ['97','98','99']):
                            print file2name
                            rfile1.Add(file2name)
            pred.append(root_numpy.tree2array(rfile1, branches = listbranch1))


f = ROOT.TFile("Ensemble.root", "recreate")
for n in range(0,len(filelist)):
    disc = pred[n]['prob_isBB']+pred[n]['prob_isB']+pred[n]['prob_isLeptB']
    x_temp, y_temp, auc = spit_out_roc(truth, disc, str(z), 'blue', flavour = True)
    x_temp1, y_temp1, auc1 = spit_out_roc(truth, disc, str(z), 'blue', flavour = False)
    gr1 = TGraph( 100, x_temp, y_temp )
    gr1.SetName("bvsl"+str(n))
    gr2 = TGraph( 100, x_temp1, y_temp1)
    gr2.SetName("bvsc"+str(n))
    gr1.Write()
    gr2.Write()
    f.Write()
    coeff = np.append(coeff,[auc])

coeff = coeff* (1/((np.max(coeff)-np.min(coeff))/0.5))
coeff = coeff-np.min(coeff)
print coeff
meantag = np.average( np.array([ pred[z]['prob_isBB']+pred[z]['prob_isB']+pred[z]['prob_isLeptB'] for z in range(0,len(filelist)) ]), axis=0, weights=coeff )
plt.figure(figsize=(18,10))

temp1, temp2, dum_ = spit_out_roc(truth, meantag, str(z), 'blue', flavour = True)
temp3, temp4, dum_ = spit_out_roc(truth, meantag, str(z), 'red', flavour = False)

f = ROOT.TFile("Ensemble.root", "update")
gr1 = TGraph( 100, temp1, temp2 )
gr1.SetName("ensemble_0")
gr2 = TGraph( 100, temp3, temp4 )
gr2.SetName("ensemble_1")
gr1.Write()
gr2.Write()
f.Write()
