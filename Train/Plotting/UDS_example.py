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


makeROCs_async('/eos/user/e/ebols/PermInvariantPrediction2/tree_association.txt',         
               name_list=['UDS vs g', 'UDS vs. CB'],         
               probabilities_list=['prob_isUDS','prob_isUDS'], 
               truths_list=['isUD+isS','isUD+isS'],        
               vetos_list=['isG','isC+isGCC+isCC+isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],         
               colors_list='auto',        
               outpdffile='PermInv_TTBarHad_UDSTag_30GeV.pdf',         
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
