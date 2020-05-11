
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


makeROCs_async('/eos/user/e/ebols/Prediction_FinalDeepFlavour2_10XTTBar/tree_association.txt',         
               name_list=['C vs. light full', 'C vs. B full'],         
               probabilities_list=['pfDeepCSVJetTags_probc/(pfDeepCSVJetTags_probc+pfDeepCSVJetTags_probudsg)','pfDeepCSVJetTags_probc/(pfDeepCSVJetTags_probc+pfDeepCSVJetTags_probudsg)'],
               #probabilities_list=['prob_isB+prob_isBB','prob_isB+prob_isBB'], 
               truths_list=['isC+isGCC+isCC','isC+isGCC+isCC'],
               vetos_list=['isUD+isS+isG','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'], 
               colors_list='auto',        
               outpdffile='DeepCSV_CTag_TTBarHad_30pt_cvsl.pdf',         
               cuts='jet_pt>30 & pfDeepCSVJetTags_probudsg > 0',            
               cmsstyle=False,     
               firstcomment='',    
               secondcomment='',   
               invalidlist='',     
               extralegend=None,   
               logY=True,          
               individual=False,   
               xaxis="b efficiency",           
               nbins=200)          
