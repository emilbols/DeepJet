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
#files = ['0p005','0p01','0p02','0p03','0p05','0p10','0p15','0p20','0p30','0p50']
files = ['0p75']
for z in files:
    makeROCs_async('/data/ml/ebols/Prediction' + z + '/tree_association.txt',         
                   name_list=['B vs light', 'B vs. C'],         
                   probabilities_list=['prob_isB+prob_isBB','prob_isB+prob_isBB'], 
                   truths_list=['isB+isGBB+isBB+isLeptonicB+isLeptonicB_C','isB+isGBB+isBB+isLeptonicB+isLeptonicB_C'],        
                   vetos_list=['isUD+isS+isG','isC+isGCC+isCC'],         
                   colors_list='auto',        
                   outpdffile='ROC'+z+'.pdf',         
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
