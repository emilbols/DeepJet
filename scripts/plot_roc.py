print("start import")
import os
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import root_numpy
import ROOT
from ROOT import TCanvas, TGraph, TGraphAsymmErrors, TH2F, TH1F
print("finish import")
from root_numpy import fill_hist


def spit_out_roc(disc,truth_array,selection_array):

    newx = np.logspace(-3.5, 0, 100)
    tprs = pd.DataFrame()
    truth = truth_array[selection_array]*1
    disc = disc[selection_array]
    tmp_fpr, tmp_tpr, _ = roc_curve(truth, disc)
    coords = pd.DataFrame()
    coords['fpr'] = tmp_fpr
    coords['tpr'] = tmp_tpr
    clean = coords.drop_duplicates(subset=['fpr'])
    spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
    tprs = spline(newx)
    return tprs, newx



pred = []
isDeepJet = False
isDeepCSV = False
isDeepJet_puppi = True
if isDeepJet:
    listbranch = ['prob_isB', 'prob_isBB','prob_isLeptB', 'prob_isC','prob_isUDS','prob_isG','isB', 'isBB', 'isLeptB', 'isC','isUDS','isG','jet_pt', 'jet_eta']
if isDeepJet_puppi:
    listbranch = ['prob_isB', 'prob_isBB','prob_isLeptB', 'prob_isC','prob_isUDS','prob_isG','prob_isPU','isB', 'isBB', 'isLeptB', 'isC','isUDS','isG','isPU','jet_pt', 'jet_eta']
if isDeepCSV:
    listbranch = ['prob_isB', 'prob_isBB', 'prob_isC','prob_isUDSG','isB', 'isBB', 'isC','isUDSG','jet_pt', 'jet_eta']

dirz = '/data/ml/ebols/DeepJet_puppi_miniTest_TTBarHad/'
truthfile = open( dirz+'outfiles.txt','r')
print("opened text file")
rfile1 = ROOT.TChain("tree")
count = 0

for line in truthfile:
    count += 1
    if len(line) < 1: continue
    file1name=str(dirz+line.split('\n')[0])
    rfile1.Add(file1name)

print("added files")
df = root_numpy.tree2array(rfile1, branches = listbranch)
print("converted to root")

if isDeepJet:
    b_jets = df['isB']+df['isBB']+df['isLeptB']
    disc = df['prob_isB']+df['prob_isBB']+df['prob_isLeptB']
    summed_truth = df['isB']+df['isBB']+df['isLeptB']+df['isC']+df['isUDS']+df['isG']
    veto_c = (df['isC'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)
    veto_udsg = (df['isUDS'] != 1) & (df['isG'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)
if isDeepJet_puppi:
    b_jets = df['isB']+df['isBB']+df['isLeptB']
    disc = df['prob_isB']+df['prob_isBB']+df['prob_isLeptB']
    pu_disc = df['prob_isPU']
    pu = df['isPU']

    summed_truth = df['isB']+df['isBB']+df['isLeptB']+df['isC']+df['isUDS']+df['isG']+df['isPU']

    veto_c = (df['isC'] != 1) & (df['isPU'] != 1) & ( df['jet_pt'] > 30) & ( np.abs(df['jet_eta']) < 2.5) & (summed_truth != 0)
    veto_udsg = (df['isUDS'] != 1) & (df['isG'] != 1) & (df['isPU'] != 1) & ( df['jet_pt'] > 30) & ( np.abs(df['jet_eta']) < 2.5) & (summed_truth != 0)

    veto_c_higheta = (df['isC'] != 1) & (df['isPU'] != 1) & ( df['jet_pt'] > 30) & ( np.abs(df['jet_eta']) > 2.5) & (summed_truth != 0)
    veto_udsg_higheta = (df['isUDS'] != 1) & (df['isG'] != 1) & (df['isPU'] != 1) & ( df['jet_pt'] > 30) & ( np.abs(df['jet_eta']) > 2.5) & (summed_truth != 0)

    pu_id_select = ( df['jet_pt'] > 10) & (df['jet_pt'] < 50) & ( np.abs(df['jet_eta']) < 2.5) & (summed_truth != 0)
    pu_id_select_higheta = ( df['jet_pt'] > 10) & (df['jet_pt'] < 50) & ( np.abs(df['jet_eta']) > 2.5) & (summed_truth != 0)

if isDeepCSV:
    b_jets = df['isB']+df['isBB']
    disc = df['prob_isB']+df['prob_isBB']
    summed_truth = df['isB']+df['isBB']+df['isC']+df['isUDSG']
    veto_c = (df['isC'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)
    veto_udsg = (df['isUDSG'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)


f = ROOT.TFile("ROCS_DeepJetPuppi.root", "recreate")
x1, y1 = spit_out_roc(disc,b_jets,veto_c)
x2, y2 = spit_out_roc(disc,b_jets,veto_udsg)
gr1 = TGraph( 100, x1, y1 )
gr1.SetName("roccurve_0")
gr2 = TGraph( 100, x2, y2 )
gr2.SetName("roccurve_1")
gr1.Write()
gr2.Write()
f.Write()
f.Close()

if isDeepJet_puppi:
    f = ROOT.TFile("ROCS_DeepJetPuppi_higheta.root", "recreate")
    x1, y1 = spit_out_roc(disc,b_jets,veto_c_higheta)
    x2, y2 = spit_out_roc(disc,b_jets,veto_udsg_higheta)
    gr1 = TGraph( 100, x1, y1 )
    gr1.SetName("roccurve_0")
    gr2 = TGraph( 100, x2, y2 )
    gr2.SetName("roccurve_1")
    gr1.Write()
    gr2.Write()
    f.Write()
    f.Close()
    f = ROOT.TFile("ROCS_DeepJetPuppi_PUID.root", "recreate")
    x1, y1 = spit_out_roc(1.0-pu_disc,1-pu,pu_id_select)
    gr1 = TGraph( 100, x1, y1 )
    gr1.SetName("roccurve_0")
    gr1.Write()
    f.Write()
    f = ROOT.TFile("ROCS_DeepJetPuppi_PUID_higheta.root", "recreate")
    x1, y1 = spit_out_roc(1.0-pu_disc,1-pu,pu_id_select_higheta)
    gr1 = TGraph( 100, x1, y1 )
    gr1.SetName("roccurve_0")
    gr1.Write()
    f.Write()
