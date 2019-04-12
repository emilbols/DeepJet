'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainDataDeepJetTest import TrainData_Flavours, TrainData_simplerTruth, fileTimeOut
import numpy as np


class TrainData_test(TrainData_Flavours, TrainData_simplerTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavours.__init__(self)
        
        self.addBranches(['Jet_uncorrpt', 'Jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal'],
                             6)
        
        
        self.addBranches(['TagVarCSV_trackEtaRel'],4)

        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dVal', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_test, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


