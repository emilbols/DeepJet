
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


makeROCs_async('/eos/user/e/ebols/Prediction_TTBarLept94X_LRPSkimV2_cuttracks_v2/tree_association.txt',         
               name_list=['B vs. light full', 'B vs. C full'],         
               probabilities_list=['pfDeepCSVJetTags_probbb+pfDeepCSVJetTags_probb','pfDeepCSVJetTags_probbb+pfDeepCSVJetTags_probb'], 
               truths_list=['isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],
               vetos_list=['isUD+isS+isG','isC+isGCC+isCC'], 
               colors_list='auto',        
               outpdffile='DeepCSV_94X_TTBarLep_30GeV.pdf',         
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