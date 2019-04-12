'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainDataDeepJet import TrainData_Flavour, TrainData_simpleTruth, TrainData_fullTruth, fileTimeOut
import numpy as np


class TrainData_deepJEC(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['gen_pt_WithNu', 'jet_eta', 'nCpfcand','nNpfcand','npv','nsv'])
       
        self.addBranches(['Cpfcan_pt','Cpfcan_BtagPf_trackJetDistVal','Cpfcan_BtagPf_trackJetDistSig','Cpfcan_BtagPf_trackDeltaR'],
                             10)
        
        self.addBranches(['Npfcan_pt','Npfcan_deltaR','Npfcan_isGamma','Npfcan_HadFrac'],10)

        self.addBranches(['muons_pt','muons_relEta','muons_relPhi'],1)

        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)

        self.registerBranches=['jet_pt']
        self.regressiontargetclasses=['uncPt','Pt']

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_deepJEC, self).readFromRootFile(filename, TupleMeanStd, weighter)

        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)

        Tuple = self.readTreeFromRootToTuple(filename)
        reg_truth=Tuple['gen_pt_WithNu'].view(np.ndarray)
        reco_pt=Tuple['jet_pt'].view(np.ndarray)

        correctionfactor=np.zeros(self.nsamples)
        for i in range(self.nsamples):
            correctionfactor[i]=reg_truth[i]/reco_pt[i]

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            undef=Tuple['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=np.empty(self.nsamples)
            weights.fill(1.)
    
        self.x = [self.x[0][mask]]
        self.y=[correctionfactor]
     
