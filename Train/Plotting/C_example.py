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


makeROCs_async('/afs/cern.ch/work/e/ebols/private/Prediction_DFSkim_NoAss_10XTTBarHad/tree_association.txt',         
               name_list=['C vs light', 'C vs. B'],         
               probabilities_list=['prob_isC','prob_isC'], 
               truths_list=['isC+isGCC+isCC','isC+isGCC+isCC'],        
               vetos_list=['isUD+isS+isG','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],         
               colors_list='auto',        
               outpdffile='DF_NoAss_TTBarHad_CTag_30GeV.pdf',         
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
