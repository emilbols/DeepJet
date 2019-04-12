
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


makeROCs_async('/afs/cern.ch/work/e/ebols/public/PredictionsDeepCSVMoreStats/tree_association.txt',         
               name_list=['B vs light DeepCSV', 'B vs. C DeepCSV'],         
               probabilities_list=['prob_isB+prob_isBB','prob_isB+prob_isBB'], 
               truths_list=['isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],        
               vetos_list=['isUD+isS+isG','isC+isGCC+isCC'],         
               colors_list='auto',        
               outpdffile='DeepCSV_noTrackSelect.pdf',         
               cuts='jet_pt>30 && prob_isB > 0',            
               cmsstyle=False,     
               firstcomment='',    
               secondcomment='',   
               invalidlist='',     
               extralegend=None,   
               logY=True,          
               individual=False,   
               xaxis="b efficiency",           
               nbins=200)          
