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


makeROCs_async('/eos/user/e/ebols/Predictors/PredictionQCD10Xon94X/tree_association.txt',         
               name_list=['UDS vs g', 'UDS vs g QGL'],         
               probabilities_list=['prob_isUDS/(prob_isUDS+prob_isG)','jet_qgl'], 
               truths_list=['isUD+isS','isUD+isS'],        
               vetos_list=['isG','isG'],         
               colors_list='auto',        
               outpdffile='DeepJet_QuarkGluonTag_QCD_30to50GeV.pdf',         
               cuts='jet_pt>30 & jet_pt<50',            
               cmsstyle=False,     
               firstcomment='',    
               secondcomment='',   
               invalidlist='',     
               extralegend=None,   
               logY=True,          
               individual=False,   
               xaxis="c efficiency",           
               nbins=200)          
