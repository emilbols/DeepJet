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
listbranch = ['isB','isC','isBB','isGBB','isLeptonicB','isLeptonicB_C','isUD','isS','isGCC','isCC','isG','jet_pt']
listbranch1 = ['prob_isB','prob_isBB','prob_isLeptB']


def misandeff(truth,pred,flavour,wp,ptlow,pthigh):

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

    sf_mis, sf_eff, sf_mis_up, sf_mis_down, sf_eff_up, sf_eff_down = SF_calc_DF(pt,wp, ptlow, flavour)

    bjets = float(len(truth[bs]))
    lightjets = float(len(truth[not_bs]))
    tp = len( truth[(pred['prob_isB']+pred['prob_isBB']+pred['prob_isLeptB'] > wp) & ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 1)] )
    fp = len( truth[(pred['prob_isB']+pred['prob_isBB']+pred['prob_isLeptB'] > wp) & ( (truth['isB']+truth['isBB']+truth['isLeptonicB_C']+truth['isLeptonicB']+truth['isGBB']) == 0)] )       
    eff = (tp/bjets)
    mistag = (fp/lightjets)

    return eff, mistag, sf_eff, sf_mis, sf_mis_up, sf_mis_down, sf_eff_up, sf_eff_down, bjets, lightjets


def SF_calc_DF(x,wp,ptlow, flavour):

    tight_var = [0.034065559506416321, 0.031077833846211433,0.027616133913397789,0.031399231404066086, 0.034734658896923065, 0.044301345944404602, 0.050837442278862, 0.040907666087150574]
    med_var = [0.076275914907455444,0.026398291811347008,0.02534114383161068,0.02437339723110199,0.026176376268267632,0.02870459109544754,0.037160992622375488, 0.036622315645217896, 0.04215230792760849]
    loose_var = [0.021110832691192627, 0.017828520387411118, 0.019102351740002632, 0.019069747999310493, 0.020529843866825104, 0.031695902347564697, 0.036414146423339844, 0.04625384509563446]

    tight_var_c = [0.11922945827245712, 0.10877241939306259, 0.096656471490859985, 0.10989730805158615, 0.12157130241394043, 0.1550547182559967, 0.1779310405254364, 0.14317682385444641]
    med_var_c = [0.079194873571395874,0.07602342963218689, 0.073120191693305969, 0.078529126942157745, 0.086113773286342621, 0.11148297786712646, 0.10986694693565369, 0.12645691633224487]
    loose_var_c = [0.052777081727981567, 0.044571302831172943, 0.047755878418684006, 0.047674369066953659, 0.051324609667062759, 0.079239755868911743, 0.091035366058349609, 0.11563461273908615]


    if wp > 0.7:
        eff = 0.908648*( (1.+(0.00516407*x) ) / ( 1.+ (0.00564675*x) ) )
        if flavour:
            mistag = 0.952956 + 0.000569069*x- 1.88872e-06*x*x + 1.25729e-09*x*x*x
            mistag_down = (0.952956+0.000569069*x-1.88872e-06*x*x+1.25729e-09*x*x*x)*(1-(0.232956+0.000143975*x-1.66524e-07*x*x))
            mistag_up = (0.952956+0.000569069*x-1.88872e-06*x*x+1.25729e-09*x*x*x)*(1+(0.232956+0.000143975*x-1.66524e-07*x*x))
        else:
            mistag = 0.908648*( (1.+(0.00516407*x) ) / ( 1.+ (0.00564675*x) ) ) 
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

    if (wp > 0.2) & (wp < 0.7):
        eff = 0.991757*((1.+(0.0209615*x))/(1.+(0.0234962*x))) 
        if flavour:
            mistag = 1.40779-0.00094558*x+8.74982e-07*x*x-4.67814/x
            mistag_down = (1.40779-0.00094558*x+8.74982e-07*x*x-4.67814/x)*(1-(0.100661+0.000294578*x-3.2739e-07*x*x))
            mistag_up = (1.40779-0.00094558*x+8.74982e-07*x*x-4.67814/x)*(1+(0.100661+0.000294578*x-3.2739e-07*x*x)) 
        else:
            mistag = 0.991757*((1.+(0.0209615*x))/(1.+(0.0234962*x))) 
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
 
    if (wp < 0.2):
        eff = 1.04891*((1.+(0.0145976*x))/(1.+(0.0165274*x)))
        if flavour:
            mistag = 1.43763-0.000337048*x+2.22072e-07*x*x-4.85489/x
            mistag_down = (1.43763-0.000337048*x+2.22072e-07*x*x-4.85489/x)*(1-(0.0526747+8.27233e-05*x-1.12281e-07*x*x))
            mistag_up = (1.43763-0.000337048*x+2.22072e-07*x*x-4.85489/x)*(1+(0.0526747+8.27233e-05*x-1.12281e-07*x*x))
        else:
            mistag = 1.04891*((1.+(0.0145976*x))/(1.+(0.0165274*x)))
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


def lal(truth,pred,filename, rocname, flavour, low_pt):
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

    bjets = np.zeros((3,len(bins)-1))
    lightjets = np.zeros((3,len(bins)-1))
    for n in range(0,len(bins)-1):
        for z in range(0,3):
            eff_tmp, mis_tmp, sf_eff_tmp, sf_mis_tmp, sf_mis_up_tmp, sf_mis_down_tmp, sf_eff_up_tmp, sf_eff_down_tmp, bjets_tmp, lightjets_tmp = misandeff(truth,pred,flavour,wp[z],bins[n],bins[n+1])
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
    print(full_eff)
    print(full_mis)
    print(np.sum(bjets[0,:]))
    print(np.sum(lightjets[0,:]))
    print(np.sum(bjets[1,:]))
    print(np.sum(lightjets[1,:]))
    print(np.sum(bjets[2,:]))
    print(np.sum(lightjets[2,:]))
   
    exl = full_eff - full_eff_down
    eyl = full_mis - full_mis_down
    exu = full_eff_up - full_eff
    eyu = full_mis_up - full_mis
    effective_sf = full_eff/full_eff_no_cor

    f = ROOT.TFile(filename, "update")
    gr1 = TGraphAsymmErrors(3, x, y, exl, exu, eyl, eyu)
    gr1.SetName(rocname)
    gr1.Write()
    gr2 = TGraph(3, full_eff_no_cor, full_mis_no_cor)
    gr2.SetName(rocname+"noCor")
    gr2.Write()

    f.Write()
    return sf_eff, sf_mis, eff, mis, effective_sf

print("start")
truthfile = open('/eos/user/e/ebols/Predictors/PredictionDeepFlavour10Xon94XTTBar/tree_association.txt','r')
print("opened text file")
rfile2 = ROOT.TChain("deepntuplizer/tree")
rfile1 = ROOT.TChain("tree")
count = 0

for line in truthfile:
    count += 1
    if len(line) < 1: continue
    file1name=str(line.split(' ')[0])
    file2name=str(line.split(' ')[1])[:-1]
    rfile2.Add(file1name)
    rfile1.Add(file2name)
    

print("added files")
truth = root_numpy.tree2array(rfile2, branches = listbranch)
pred = root_numpy.tree2array(rfile1, branches = listbranch1)
print("converted to root")

f = ROOT.TFile("SF_deepFlavour_30GeV.root", "recreate")       
f.Write()
sf_eff, sf_mis, eff, mis, effective_sf = lal(truth,pred,"SF_deepFlavour_30GeV.root","roccurve_0",True,30)
sf_eff_bvsc, sf_mis_bvsc, eff_bvsc, mis_bvsc, effective_sf_2 = lal(truth,pred,"SF_deepFlavour_30GeV.root","roccurve_1",False,30)

print 'light jets'
print sf_eff
print sf_mis
print eff
print mis
print effective_sf

print 'c jets'
print sf_eff_bvsc
print sf_mis_bvsc
print eff_bvsc
print mis_bvsc
print effective_sf_2
