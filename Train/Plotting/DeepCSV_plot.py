
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

#/afs/cern.ch/work/e/ebols/private/PredictionQCD_3200_Inf_10X/tree_association.txt
#/data/ml/ebols/Prediction_RNN_DeepCSV_sixTracks_FullStats/
#/data/ml/ebols/Prediction_DeepCSVRNN_MT_MoreTracksReally
makeROCs_async('/data/ml/ebols/Prediction_ttBar_6TracksRNNCSV/tree_association.txt',         
               name_list=['B vs. light full', 'B vs. C full'],         
               #probabilities_list=['pfDeepCSVJetTags_probbb+pfDeepCSVJetTags_probb','pfDeepCSVJetTags_probbb+pfDeepCSVJetTags_probb'],
               probabilities_list=['prob_isB+prob_isBB','prob_isB+prob_isBB'], 
               truths_list=['isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],
               vetos_list=['isUD+isS+isG','isC+isGCC+isCC'], 
               colors_list='auto',        
               outpdffile='DeepCSV_RNN_6Tracks_redone.pdf',         
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
