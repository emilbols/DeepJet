from DeepJetCore.evaluation import makeROCs_async




#makeROCs_async(intextfile, 
#               name_list, 
#               probabilities_list, 
#               truths_list, 
#               vetos_list,
#               colors_list, 
#               outpdffile, 
#               cuts='',
#               cmsstyle=False, 
#               firstcomment='',
#               secondcomment='',
#               invalidlist='',
#               extralegend=None,
#               logY=True,
#               individual=False,
#               xaxis="",
#               nbins=200)

makeROCs_async('/afs/cern.ch/work/e/ebols/private/PredictTrainingPhase0_onPhase0_QCD120to170/tree_association.txt',         
               name_list=['B vs light DF', 'B vs. C DF','B vs light DeepCSV', 'B vs. C DeepCSV'],         
               probabilities_list=['prob_isB+prob_isBB+prob_isLeptB','prob_isB+prob_isBB+prob_isLeptB','pfDeepCSVJetTags_probbb+pfDeepCSVJetTags_probb','pfDeepCSVJetTags_probbb+pfDeepCSVJetTags_probb'], 
               truths_list=['isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],        
               vetos_list=['isUD+isS+isG','isC+isGCC+isCC','isUD+isS+isG','isC+isGCC+isCC'],         
               colors_list='auto',        
               outpdffile='DF_Phase0_on_Phase0.pdf',         
               cuts='jet_pt>30',            
               cmsstyle=False,     
               firstcomment='',    
               secondcomment='',   
               invalidlist='',     
               extralegend=None,   
               logY=True,          
               individual=False,   
               xaxis="b efficiency",           
               nbins=200)          
