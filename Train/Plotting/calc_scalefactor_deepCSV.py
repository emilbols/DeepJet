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
from ROOT import TCanvas, TGraph, TGraphAsymmErrors
print("finish import")
#,'pfDeepCSVJetTags_probb','pfDeepCSVJetTags_probbb'
listbranch = ['isB','isC','isBB','isGBB','isLeptonicB','isLeptonicB_C','isUD','isS','isGCC','isCC','isG','jet_pt','pfDeepCSVJetTags_probbb','pfDeepCSVJetTags_probb']
listbranch1 = ['prob_isB','prob_isBB']


def misandeff(truth,flavour,wp,ptlow,pthigh):

    pt = truth['jet_pt']
    if flavour:
        cs = ( (truth['isC'] == 0) & (truth['isCC'] == 0) & (truth['isGCC'] == 0) & (pt > ptlow) & (pt < pthigh) )
    else:
        cs = ( (truth['isUD'] == 0) & (truth['isS'] == 0) & (truth['isG'] == 0) & (pt > ptlow) & (pt < pthigh) )
    truth = truth[cs]
    pt = truth['jet_pt']
    not_bs = ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 0)
    bs = ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 1)

    sf_mis, sf_eff, sf_mis_up, sf_mis_down, sf_eff_up, sf_eff_down = SF_calc_DF(pt,wp, ptlow, flavour)

    bjets = float(len(truth[bs]))
    lightjets = float(len(truth[not_bs]))
    tp = len( truth[( truth['pfDeepCSVJetTags_probb']+truth['pfDeepCSVJetTags_probbb'] > wp) & ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 1)] )
    fp = len( truth[(truth['pfDeepCSVJetTags_probb']+truth['pfDeepCSVJetTags_probbb'] > wp) & ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 0)] )
    eff = (tp/bjets)
    mistag = (fp/lightjets)

    return eff, mistag, sf_eff, sf_mis, sf_mis_up, sf_mis_down, sf_eff_up, sf_eff_down, bjets, lightjets


def SF_calc_DF(x,wp,ptlow, flavour):

    tight_var = [0.017260266467928886, 0.014377081766724586, 0.014780527912080288, 0.015602967701852322, 0.018584491685032845, 0.031037848442792892, 0.038253787904977798, 0.05817108228802681]
    med_var = [0.015137125737965107, 0.013977443799376488, 0.012607076205313206, 0.013979751616716385, 0.015011214651167393, 0.034551065415143967, 0.040168888866901398, 0.054684814065694809]
    loose_var = [0.013658319599926472, 0.013741940259933472, 0.012469016946852207, 0.012925124727189541, 0.014544503763318062, 0.024242779240012169, 0.040270473808050156, 0.06284312903881073]

    tight_var_c = [0.060410931706428528, 0.050319787114858627, 0.05173184722661972, 0.054610386490821838, 0.065045721828937531, 0.10863246768712997, 0.13388825953006744, 0.20359878242015839]
    med_var_c = [0.045411378145217896, 0.041932329535484314, 0.037821229547262192, 0.041939254850149155, 0.045033644884824753, 0.1036531925201416, 0.12050666660070419, 0.16405443847179413]
    loose_var_c = [0.034145798534154892, 0.034354850649833679, 0.031172541901469231, 0.03231281042098999, 0.03636125847697258, 0.060606949031352997, 0.10067618638277054, 0.15710783004760742]


    if wp > 0.7:
        eff = 0.9201*((1.+(0.0115429*x))/(1.+(0.0119144*x))) 
        if flavour:
            mistag = 0.744235+0.959064/np.sqrt(x)
            mistag_down = (0.744235+0.959064/np.sqrt(x))*(1-(0.223641+0.000121598*x-7.65779e-08*x*x))
            mistag_up = (0.744235+0.959064/np.sqrt(x))*(1+(0.223641+0.000121598*x-7.65779e-08*x*x)) 
        else:
            mistag = 0.9201*((1.+(0.0115429*x))/(1.+(0.0119144*x))) 
            if 29 <= ptlow <= 49:
                mistag_up = mistag+tight_var_c[0]
                mistag_down = mistag-tight_var_c[0]
            if 49 <= ptlow <= 69:
                mistag_up = mistag+tight_var_c[1]
                mistag_down = mistag-tight_var_c[1]
            if 69 <= ptlow <= 99:
                mistag_up = mistag+tight_var_c[2]            
                mistag_down = mistag-tight_var_c[2]
            if 99 <= ptlow <= 139:
                mistag_up = mistag+tight_var_c[3]
                mistag_down = mistag-tight_var_c[3]
            if 139 <= ptlow <= 199:
                mistag_up = mistag+tight_var_c[4]
                mistag_down = mistag-tight_var_c[4]
            if 199 <= ptlow <= 299:
                mistag_up = mistag+tight_var_c[5]            
                mistag_down = mistag-tight_var_c[5]
            if 299 <= ptlow <= 599:
                mistag_up = mistag+tight_var_c[6]
                mistag_down = mistag-tight_var_c[6]
            if 599 <= ptlow <= 999:
                mistag_up = mistag+tight_var_c[7]
                mistag_down = mistag-tight_var_c[7]
 
        if 29 <= ptlow <= 49:
            eff_up = eff+tight_var[0]
            eff_down = eff-tight_var[0]
        if 49 <= ptlow <= 69:
            eff_up = eff+tight_var[1]
            eff_down = eff-tight_var[1]
        if 69 <= ptlow <= 99:
            eff_up = eff+tight_var[2]            
            eff_down = eff-tight_var[2]
        if 99 <= ptlow <= 139:
            eff_up = eff+tight_var[3]
            eff_down = eff-tight_var[3]
        if 139 <= ptlow <= 199:
            eff_up = eff+tight_var[4]
            eff_down = eff-tight_var[4]
        if 199 <= ptlow <= 299:
            eff_up = eff+tight_var[5]            
            eff_down = eff-tight_var[5]
        if 299 <= ptlow <= 599:
            eff_up = eff+tight_var[6]
            eff_down = eff-tight_var[6]
        if 599 <= ptlow <= 999:
            eff_up = eff+tight_var[7]
            eff_down = eff-tight_var[7]

    if (wp > 0.3) & (wp < 0.7):
        eff = 2.22144*((1.+(0.540134*x))/(1.+(1.30246*x))) 
        if flavour:
            mistag = 0.972902+0.000201811*x+3.96396e-08*x*x-4.53965e-10*x*x*x 
            mistag_down = (0.972902+0.000201811*x+3.96396e-08*x*x-4.53965e-10*x*x*x)*(1-(0.101236+0.000212696*x-1.71672e-07*x*x)) 
            mistag_up = (0.972902+0.000201811*x+3.96396e-08*x*x-4.53965e-10*x*x*x)*(1+(0.101236+0.000212696*x-1.71672e-07*x*x)) 
        else:
            mistag = 2.22144*((1.+(0.540134*x))/(1.+(1.30246*x)))  
            if 29 <= ptlow <= 49:
                mistag_up = mistag+med_var_c[0]
                mistag_down = mistag-med_var_c[0]
            if 49 <= ptlow <= 69:
                mistag_up = mistag+med_var_c[1]
                mistag_down = mistag-med_var_c[1]
            if 69 <= ptlow <= 99:
                mistag_up = mistag+med_var_c[2]            
                mistag_down = mistag-med_var_c[2]
            if 99 <= ptlow <= 139:
                mistag_up = mistag+med_var_c[3]
                mistag_down = mistag-med_var_c[3]
            if 139 <= ptlow <= 199:
                mistag_up = mistag+med_var_c[4]
                mistag_down = mistag-med_var_c[4]
            if 199 <= ptlow <= 299:
                mistag_up = mistag+med_var_c[5]            
                mistag_down = mistag-med_var_c[5]
            if 299 <= ptlow <= 599:
                mistag_up = mistag+med_var_c[6]
                mistag_down = mistag-med_var_c[6]
            if 599 <= ptlow <= 999:
                mistag_up = mistag+med_var_c[7]
                mistag_down = mistag-med_var_c[7]

        if 29 <= ptlow <= 49:
            eff_up = eff+med_var[0]
            eff_down = eff-med_var[0]
        if 49 <= ptlow <= 69:
            eff_up = eff+med_var[1]
            eff_down = eff-med_var[1]
        if 69 <= ptlow <= 99:
            eff_up = eff+med_var[2]            
            eff_down = eff-med_var[2]
        if 99 <= ptlow <= 139:
            eff_up = eff+med_var[3]
            eff_down = eff-med_var[3]
        if 139 <= ptlow <= 199:
            eff_up = eff+med_var[4]
            eff_down = eff-med_var[4]
        if 199 <= ptlow <= 299:
            eff_up = eff+med_var[5]            
            eff_down = eff-med_var[5]
        if 299 <= ptlow <= 599:
            eff_up = eff+med_var[6]
            eff_down = eff-med_var[6]
        if 599 <= ptlow <= 999:
            eff_up = eff+med_var[7]
            eff_down = eff-med_var[7]
 
    if (wp < 0.3):
        eff = 1.0942+(-(0.00468151*(np.log(x+19)*(np.log(x+18)*(3-(0.365115*np.log(x+18))))))) 
        if flavour:
            mistag = 1.07073+0.000128481*x+6.16477e-07*x*x-5.65803e-10*x*x*x
            mistag_down = (1.07073+0.000128481*x+6.16477e-07*x*x-5.65803e-10*x*x*x)*(1-(0.0485052+3.93839e-05*x-4.90281e-08*x*x))
            mistag_up = (1.07073+0.000128481*x+6.16477e-07*x*x-5.65803e-10*x*x*x)*(1+(0.0485052+3.93839e-05*x-4.90281e-08*x*x)) 
        else:
            mistag = 1.0942+(-(0.00468151*(np.log(x+19)*(np.log(x+18)*(3-(0.365115*np.log(x+18)))))))
            if 29 <= ptlow <= 49:
                mistag_up = mistag+loose_var_c[0]
                mistag_down = mistag-loose_var_c[0]
            if 49 <= ptlow <= 69:
                mistag_up = mistag+loose_var_c[1]
                mistag_down = mistag-loose_var_c[1]
            if 69 <= ptlow <= 99:
                mistag_up = mistag+loose_var_c[2]            
                mistag_down = mistag-loose_var_c[2]
            if 99 <= ptlow <= 139:
                mistag_up = mistag+loose_var_c[3]
                mistag_down = mistag-loose_var_c[3]
            if 139 <= ptlow <= 199:
                mistag_up = mistag+loose_var_c[4]
                mistag_down = mistag-loose_var_c[4]
            if 199 <= ptlow <= 299:
                mistag_up = mistag+loose_var_c[5]            
                mistag_down = mistag-loose_var_c[5]
            if 299 <= ptlow <= 599:
                mistag_up = mistag+loose_var_c[6]
                mistag_down = mistag-loose_var_c[6]
            if 599 <= ptlow <= 999:
                mistag_up = mistag+loose_var_c[7]
                mistag_down = mistag-loose_var_c[7]

        if 29 <= ptlow <= 49:
            eff_up = eff+loose_var[0]
            eff_down = eff-loose_var[0]
        if 49 <= ptlow <= 69:
            eff_up = eff+loose_var[1]
            eff_down = eff-loose_var[1]
        if 69 <= ptlow <= 99:
            eff_up = eff+loose_var[2]            
            eff_down = eff-loose_var[2]
        if 99 <= ptlow <= 139:
            eff_up = eff+loose_var[3]
            eff_down = eff-loose_var[3]
        if 139 <= ptlow <= 199:
            eff_up = eff+loose_var[4]
            eff_down = eff-loose_var[4]
        if 199 <= ptlow <= 299:
            eff_up = eff+loose_var[5]            
            eff_down = eff-loose_var[5]
        if 299 <= ptlow <= 599:
            eff_up = eff+loose_var[6]
            eff_down = eff-loose_var[6]
        if 599 <= ptlow <= 999:
            eff_up = eff+loose_var[7]
            eff_down = eff-loose_var[7]

    return np.mean(mistag), np.mean(eff), np.mean(mistag_up), np.mean(mistag_down), np.mean(eff_up), np.mean(eff_down)


def lal(truth,filename, rocname, flavour, low_pt):
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
    #wp = [0.2219,0.6324,0.8958] 
    wp = [0.1522,0.4941,0.8001]
    bjets = np.zeros((3,len(bins)-1))
    lightjets = np.zeros((3,len(bins)-1))
    for n in range(0,len(bins)-1):
        for z in range(0,3):
            eff_tmp, mis_tmp, sf_eff_tmp, sf_mis_tmp, sf_mis_up_tmp, sf_mis_down_tmp, sf_eff_up_tmp, sf_eff_down_tmp, bjets_tmp, lightjets_tmp = misandeff(truth,flavour,wp[z],bins[n],bins[n+1])
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

    f = ROOT.TFile(filename, "update")
    gr1 = TGraphAsymmErrors(3, x, y, exl, exu, eyl, eyu)
    gr1.SetName(rocname)
    gr1.Write()
    f.Write()
    gr2 = TGraph(3, full_eff_no_cor, full_mis_no_cor)
    gr2.SetName(rocname+"noCor")
    gr2.Write()
    return sf_eff, sf_mis, eff, mis, effective_sf


print("start")
#truthfile = open('/eos/user/e/ebols/Predictors/PredictionDeepFlavour10Xon94XTTBar/tree_association.txt','r')
#truthfile = open('/eos/user/e/ebols/Predictors/PredictionTTBar/tree_association.txt','r')
#truthfile = open('/data/ml/ebols/DeepCSV_ActualCMSSW_ttbar94X_fall17/tree_association.txt','r')
truthfile = open('/eos/user/e/ebols/DeepCSVRemade/Fri_181028_DeepCSVRemade/ntuple_ttbar_had_94X_remade_DeepCSV/output/less_stats.txt','r')

print("opened text file")
rfile2 = ROOT.TChain("deepntuplizer/tree")
rfile1 = ROOT.TChain("tree")
count = 0


for line in truthfile:
    count += 1
    if len(line) < 1: continue
    file1name=str(line.split(' ')[0])[:-1]
    #file2name=str(line.split(' ')[1])[:-1]
    rfile2.Add(file1name)
    #rfile1.Add(file2name)
    print(file1name)

print("added files")
truth1 = root_numpy.tree2array(rfile2, branches = listbranch)
#pred1 = root_numpy.tree2array(rfile1, branches = listbranch1)
print("converted to root")

f = ROOT.TFile("SF_deepCSV_phase1_150GeV.root", "recreate")       
f.Write()
sf_eff, sf_mis, eff, mis, effective_sf = lal(truth1,"SF_deepCSV_phase1_150GeV.root","roccurve_0",True,150)
sf_eff_bvsc, sf_mis_bvsc, eff_bvsc, mis_bvsc, effective_sf_same = lal(truth1,"SF_deepCSV_phase1_150GeV.root","roccurve_1",False,150)
#SF_miswp2, SF_effwp2 = SF_tight(jets)
#SF_miswp1, SF_effwp1 = SF_mid(jets)
#SF_miswp0, SF_effwp0 = SF_loose(jets)


#avg_SFefftight = np.mean(SF_effwp2)
#avg_SFeffmid = np.mean(SF_effwp1)
#avg_SFeffloose = np.mean(SF_effwp0)


#avg_SFmistight = np.mean(SF_miswp2)
#avg_SFmismid = np.mean(SF_miswp1)
#avg_SFmisloose = np.mean(SF_miswp0)


#file = open('testfile.txt','w')
#file.write("scale factor for eff, loose, mid , tight")
#file.write(avg_SFeffloose)
#file.write(avg_SFeffmid)
#file.write(avg_SFefftight)
#file.write("scale factor for mis, loose, mid , tight")
#file.write(avg_SFmisloose)
#file.write(avg_SFmismid)
#file.write(avg_SFmistight)
#file.close()

#for cjets
#        if 29 <= ptlow <= 49:
 #           eff_up = 
  #          eff_down = eff-0.11922945827245712
   #     if 49 <= ptlow <= 69:
    #        eff_down = eff-0.10877241939306259
     #   if 69 <= ptlow <= 99:
     #       eff_down = eff-0.096656471490859985 
     #   if 99 <= ptlow <= 139:
     #       eff_down = eff-0.10989730805158615
     #   if 139 <= ptlow <= 199:
     #       eff_down = eff-0.12157130241394043
     #   if 199 <= ptlow <= 299:
     #       eff_down = eff-0.1550547182559967
     #   if 299 <= ptlow <= 599:
     #       eff_down = eff-0.1779310405254364
     #   if 599 <= ptlow <= 999:
     #       eff_down = eff-0.14317682385444641
