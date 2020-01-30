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
               name_list=['C vs light', 'C vs. B','DeepCSV C vs light', 'DeepCSV C vs. B'],         
               probabilities_list=['prob_isC/(prob_isC+prob_isUDS+prob_isG)','prob_isC/(prob_isC+prob_isB+prob_isBB+prob_isLeptB)','pfDeepCSVJetTags_probc/(pfDeepCSVJetTags_probc+pfDeepCSVJetTags_probudsg)','pfDeepCSVJetTags_probc/(pfDeepCSVJetTags_probc+pfDeepCSVJetTags_probb+pfDeepCSVJetTags_probbb)'], 
               truths_list=['isC+isGCC+isCC','isC+isGCC+isCC','isC+isGCC+isCC','isC+isGCC+isCC'],        
               vetos_list=['isUD+isS+isG','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isUD+isS+isG','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],         
               colors_list='auto',        
               outpdffile='DeepFlavour_TTBarHad_CTag_30GeV_badd.pdf',         
               cuts='jet_pt>30',            
               cmsstyle=False,     
               firstcomment='',    
               secondcomment='',   
               invalidlist='',     
               extralegend=None,   
               logY=True,          
               individual=False,   
               xaxis="c efficiency",           
               nbins=200)          
