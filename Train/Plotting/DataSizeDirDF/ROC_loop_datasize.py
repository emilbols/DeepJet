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

dirs = ['frac0p02','frac0p05','frac0p10','frac0p15','frac0p20','frac0p30','frac0p50']
for i in dirs:
    makeROCs_async('/data/ml/ebols/Prediction_FinalDeepFlavour_'+i+'_10XTTBar/tree_association.txt',         
                   name_list=['B vs light', 'B vs. C'],         
                   probabilities_list=['prob_isB+prob_isBB+prob_isLeptB','prob_isB+prob_isBB+prob_isLeptB'], 
                   truths_list=['isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],        
                   vetos_list=['isUD+isS+isG','isC+isGCC+isCC'],         
                   colors_list='auto',        
                   outpdffile='DF_'+i+'_Skim_Final_ttBarHad_200.pdf',         
                   cuts='jet_pt>200',            
                   cmsstyle=False,     
                   firstcomment='',    
                   secondcomment='',   
                   invalidlist='',     
                   extralegend=None,   
                   logY=True,          
                   individual=False,   
                   xaxis="b efficiency",           
                   nbins=200)          
