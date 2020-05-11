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
from numpy import log
import root_numpy
import ROOT
from numpy import sqrt
from ROOT import TCanvas, TGraph, TGraphAsymmErrors
print("finish import")

listbranch = ['isB','isC','isBB','isGBB','isLeptonicB','isLeptonicB_C','isUD','isS','isGCC','isCC','isG','jet_pt','pfDeepCSVJetTags_probbb','pfDeepCSVJetTags_probb']
listbranch1 = ['prob_isB','prob_isBB','prob_isLeptB']

def calc_mistag(x,wp_sf):
    exec("sf_mis = "+wp_sf[wp_sf.sysType == 'central']['formula '].values[-1])
    exec("sf_mis_up = "+wp_sf[wp_sf.sysType == 'up']['formula '].values[-1])
    exec("sf_mis_down = "+wp_sf[wp_sf.sysType == 'down']['formula '].values[-1])
    return sf_mis, sf_mis_up, sf_mis_down

def calc_eff(x,wp_sf,ptBin):
    up_sf = wp_sf[wp_sf.sysType == 'up']
    up_sf = up_sf[(up_sf.ptMin <= ptBin) & (up_sf.ptMax > ptBin)]
    down_sf = wp_sf[wp_sf.sysType == 'down']
    down_sf = down_sf[(down_sf.ptMin <= ptBin) & (down_sf.ptMax > ptBin)]
    exec("sf_eff = "+wp_sf[wp_sf.sysType == 'central']['formula '].values[-1])
    exec("sf_eff_up = "+up_sf['formula '].values[-1])
    exec("sf_eff_down = "+down_sf['formula '].values[-1])
    return sf_eff, sf_eff_up, sf_eff_down

def misandeff(truth,pred,flavour,wp,ptlow,pthigh, SF, wp_int,tagger):

    pt = truth['jet_pt']
    if flavour:
        cs = ( (truth['isC'] == 0) & (truth['isCC'] == 0) & (truth['isGCC'] == 0) & (pt > ptlow) & (pt < pthigh) )
    else:
        cs = ( (truth['isUD'] == 0) & (truth['isS'] == 0) & (truth['isG'] == 0) & (pt > ptlow) & (pt < pthigh) )
    truth = truth[cs]
    pred = pred[cs]
    pt = truth['jet_pt']
    not_bs = ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 0)
    bs = ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 1)

    sf_mis, sf_eff, sf_mis_up, sf_mis_down, sf_eff_up, sf_eff_down = SF_calc_DF(pt,wp_int, ptlow, flavour, SF)

    bjets = float(len(truth[bs]))
    lightjets = float(len(truth[not_bs]))
    if tagger == 'DeepCSV':
        disc = truth['pfDeepCSVJetTags_probb']+truth['pfDeepCSVJetTags_probbb']
    else:
        disc = pred['prob_isB']+pred['prob_isBB']+pred['prob_isLeptB']

    tp = len( truth[(disc > wp) & ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 1)] )
    fp = len( truth[(disc > wp) & ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 0)] )       
    eff = (tp/bjets)
    mistag = (fp/lightjets)

    return eff, mistag, sf_eff, sf_mis, sf_mis_up, sf_mis_down, sf_eff_up, sf_eff_down, bjets, lightjets


def SF_calc_DF(x,wp_int,ptlow, flavour, SF):
    
    b_sfs = SF[(SF.measurementType == 'comb') & (SF.jetFlavor == 0)]
    c_sfs = SF[(SF.measurementType == 'comb') & (SF.jetFlavor == 1)]
    udsg_sfs = SF[(SF.measurementType == 'incl') & (SF.jetFlavor == 2)]

    b_sfs = b_sfs[b_sfs.OperatingPoint == wp_int]
    c_sfs = c_sfs[c_sfs.OperatingPoint == wp_int]
    udsg_sfs = udsg_sfs[udsg_sfs.OperatingPoint == wp_int]


    if flavour:
        sf_mis, sf_mis_up, sf_mis_down = calc_mistag(x,udsg_sfs)
    else:
        sf_mis, sf_mis_up, sf_mis_down = calc_eff(x,c_sfs, ptlow)
    
    sf_eff, sf_eff_up, sf_eff_down = calc_eff(x,b_sfs,ptlow)

    return np.mean(sf_mis), np.mean(sf_eff), np.mean(sf_mis_up), np.mean(sf_mis_down), np.mean(sf_eff_up), np.mean(sf_eff_down)


def lal(truth,pred,filename, rocname, flavour, low_pt, SF_file,tagger):
    lowbins = range(low_pt,500,10)
    highbins = range(500,1100,100)
    bins = lowbins+highbins
    eff = np.zeros((3,len(bins)-1))
    mis = np.zeros((3,len(bins)-1))
    sf_eff = np.zeros((3,len(bins)-1))
    sf_mis = np.zeros((3,len(bins)-1))
    sf_eff_up = np.zeros((3,len(bins)-1))
    sf_mis_up = np.zeros((3,len(bins)-1))
    sf_eff_down = np.zeros((3,len(bins)-1))
    sf_mis_down = np.zeros((3,len(bins)-1))
    wp = [0.0521,0.3033,0.7489] 
    SF = pd.read_csv(SF_file,skipinitialspace=True)
    bjets = np.zeros((3,len(bins)-1))
    lightjets = np.zeros((3,len(bins)-1))
    for n in range(0,len(bins)-1):
        for z in range(0,3):
            eff_tmp, mis_tmp, sf_eff_tmp, sf_mis_tmp, sf_mis_up_tmp, sf_mis_down_tmp, sf_eff_up_tmp, sf_eff_down_tmp, bjets_tmp, lightjets_tmp = misandeff(truth,pred,flavour,wp[z],bins[n],bins[n+1], SF,z,tagger)
            eff[z,n] = eff_tmp
            mis[z,n] = mis_tmp
            sf_eff[z,n] = sf_eff_tmp
            sf_mis[z,n] = sf_mis_tmp
            sf_eff_up[z,n] = sf_eff_up_tmp
            sf_mis_up[z,n] = sf_mis_up_tmp
            sf_eff_down[z,n] = sf_eff_down_tmp
            sf_mis_down[z,n] = sf_mis_down_tmp
            bjets[z,n] = bjets_tmp
            lightjets[z,n] = lightjets_tmp


    crosscheck = 0
    full_eff = np.zeros(3)
    full_mis = np.zeros(3)
    full_eff_no_cor = np.zeros(3)
    full_mis_no_cor = np.zeros(3)
    full_eff_down = np.zeros(3)
    full_mis_down = np.zeros(3)
    full_eff_up = np.zeros(3)
    full_mis_up = np.zeros(3)
    bin_eff_w = 0
    bin_mis_w = 0

    for n in range(0, len(bins)-1):
        bin_eff_w = float(bjets[0,n])/np.sum(bjets[0,:])
        crosscheck += bin_eff_w
        bin_mis_w = float(lightjets[0,n])/np.sum(lightjets[0,:])
        for z in range(0, 3):
            full_eff[z] += sf_eff[z,n]*eff[z,n]*bin_eff_w
            full_mis[z] += sf_mis[z,n]*mis[z,n]*bin_mis_w
            full_eff_no_cor[z] += eff[z,n]*bin_eff_w
            full_mis_no_cor[z] += mis[z,n]*bin_mis_w
            full_eff_down[z] += eff[z,n]*sf_eff_down[z,n]*bin_eff_w
            full_mis_down[z] += sf_mis_down[z,n]*mis[z,n]*bin_mis_w
            full_eff_up[z] += eff[z,n]*sf_eff_up[z,n]*bin_eff_w
            full_mis_up[z] += mis[z,n]*sf_mis_up[z,n]*bin_mis_w

    print(crosscheck)    
    x = full_eff
    y = full_mis
   
    exl = full_eff - full_eff_down
    eyl = full_mis - full_mis_down
    exu = full_eff_up - full_eff
    eyu = full_mis_up - full_mis

    effective_sf = full_eff/full_eff_no_cor
    esf_err1 = (full_eff_up - full_eff)/full_eff_no_cor
    esf_err2 = (full_eff-full_eff_down)/full_eff_no_cor


    effective_mis = full_mis/full_mis_no_cor
    emis_err1 = (full_mis_up - full_mis)/full_mis_no_cor
    emis_err2 = (full_mis-full_mis_down)/full_mis_no_cor

    f = ROOT.TFile(filename, "update")
    gr1 = TGraphAsymmErrors(3, x, y, exl, exu, eyl, eyu)
    gr1.SetName(rocname)
    gr1.Write()
    gr2 = TGraph(3, full_eff_no_cor, full_mis_no_cor)
    gr2.SetName(rocname+"noCor")
    gr2.Write()

    f.Write()
    return sf_eff, sf_mis, eff, mis, effective_sf, esf_err1, esf_err2, effective_mis, emis_err1, emis_err2 

print("start")

year = '2018'
#year = '2017
tagger = 'DeepFlavour'
#tagger = 'DeepCSV'
if tagger == 'DeepCSV':
    truthfile = open('/eos/user/e/ebols/DeepCSVRemade/Fri_181028_DeepCSVRemade/ntuple_ttbar_had_94X_remade_DeepCSV/output/less_stats.txt','r')
else:
    truthfile = open('/eos/user/e/ebols/Predictors/PredictionDeepFlavour10Xon94XTTBar/tree_association.txt','r')

print("opened text file")

rfile2 = ROOT.TChain("deepntuplizer/tree")
rfile1 = ROOT.TChain("tree")
count = 0

for line in truthfile:
    count += 1
    if len(line) < 1: continue
    file1name=str(line.split(' ')[0])
    if tagger == 'DeepCSV':
        file2name=str(line.split('\n')[0])
        file1name = file2name
    else:
        file2name=str(line.split(' ')[1])[:-1]
    rfile2.Add(file1name)
    rfile1.Add(file2name)
    

print("added files")


truth = root_numpy.tree2array(rfile2, branches = listbranch)

if tagger == 'DeepCSV':
    pred = truth
else:
    pred = root_numpy.tree2array(rfile1, branches = listbranch1)
print("converted to root")

if tagger == 'DeepCSV':
    if year is '2017':
        SF_file = 'DeepCSV_94XSF_V4_B_F.csv'
    else:
        SF_file = 'DeepCSV_102XSF_V1.csv'
else:
    if year is '2017':
        SF_file = 'DeepFlavour_94XSF_V2_B_F.csv'
    else:
        SF_file = 'DeepJet_102XSF_V1.csv'
pt_cut = 30

filename = "automated_SF_"+tagger+"_"+str(pt_cut)+"GeV_"+year+".root"
f = ROOT.TFile(filename, "recreate")       
f.Write()
sf_eff, sf_mis, eff, mis, effective_sf, esf_err1, esf_err2, effective_mis, emis_err1, emis_err2 = lal(truth,pred,filename,"roccurve_0",True,pt_cut,SF_file,tagger)

f = ROOT.TFile(filename, "update")
if tagger is 'DeepCSV':
    height =np.array([0.0,1.0,2.0])
else:
    height =np.array([0-0.15,1-0.15,2-0.15])
gr1 = TGraphAsymmErrors(3, effective_sf, height, esf_err2, esf_err1, np.array([0,0,0]), np.array([0,0,0]))
gr1.SetName('ScaleFactors')
gr1.Write()
gr2 = TGraphAsymmErrors(3, effective_mis, height, emis_err2, emis_err1, np.array([0,0,0]), np.array([0,0,0]))
gr2.SetName('misid')
gr2.Write()
f.Write()

for n in range(0,3):
    print('WP' + str(n))
    print(str(effective_sf[n]) + ' + ' + str(esf_err1[n]) + ' - '+ str(esf_err2[n]))
    print(str(effective_mis[n]) + ' + ' + str(emis_err1[n]) + ' - '+str(emis_err2[n]))
#sf_eff_bvsc, sf_mis_bvsc, eff_bvsc, mis_bvsc, effective_sf_2 = lal(truth,pred,"automated_SF_"+tagger+"_"+str(pt_cut)+"GeV.root","roccurve_1",False,pt_cut,SF_file,tagger)


