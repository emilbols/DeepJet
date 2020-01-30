'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainDataDeepJet import TrainData_Flavour, TrainData_Flavour_noNorm, TrainData_simpleTruth, TrainData_Flavour_MCDATA, TrainData_simpleTruth_MCDATA, TrainData_fullTruth, fileTimeOut
import numpy as np


class TrainData_deepCSV(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
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
        super(TrainData_deepCSV, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]

class TrainData_deepCSV_leptons(TrainData_Flavour_noNorm, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour_noNorm.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
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
        
        self.addBranches(['electrons_pt', 
                              'electrons_relEta',
                              'electrons_relPhi',
                              'electrons_energy'],
                             2)
        
        self.addBranches(['muons_pt', 
                          'muons_relEta',
                          'muons_relPhi',
                          'muons_energy'],
                             2)


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_deepCSV_leptons, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


class TrainData_deepCSV_noNorm(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        self.remove = False
        
        self.addBranches(['jet_pt', 'jet_eta',
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
        super(TrainData_deepCSV_noNorm, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


class TrainData_deepCSV_MCDATA(TrainData_Flavour_MCDATA, TrainData_simpleTruth_MCDATA):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour_MCDATA.__init__(self)
        
        self.addBranches(['Jet_pt', 'n_PV', 'Jet_eta',
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
        super(TrainData_deepCSV_MCDATA, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


class TrainData_deepCSV_noTrackSelect(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackJetDistVal',
                              'Cpfcan_BtagPf_trackPtRel', 
                              'Cpfcan_BtagPf_trackDeltaR', 
                              'Cpfcan_BtagPf_trackPtRatio', 
                              'Cpfcan_BtagPf_trackSip3dSig', 
                              'Cpfcan_BtagPf_trackSip2dSig'],
                             6)
        
        
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel'],4)

        self.addBranches(['sv_mass', 
                              'sv_ntracks', 
                              'sv_enratio',
                              'sv_deltaR',
                              'sv_dxy', 
                              'sv_dxysig', 
                              'sv_d3d', 
                              'sv_d3dsig'],
                             1)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_deepCSV_noTrackSelect, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


class TrainData_deepCSV_random_pruned(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_jetNSelectedTracks'])
       
        self.addBranches(['TagVarCSVTrk_trackJetDistVal', 
                              'TagVarCSVTrk_trackDeltaR'],
                             6)
        
        self.addBranches(['TagVarCSVTrk_trackDecayLenVal'],
                             6)
        
        self.addBranches(['TagVarCSVTrk_trackPtRel'],
                             4)

        self.addBranches(['TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig'],
                             3)

        self.addBranches(['TagVarCSVTrk_trackSip2dSig'],
                             2)
        
        self.addBranches(['TagVarCSV_trackEtaRel'],4)

        self.addBranches(['TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal'],
                             1)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_deepCSV_random_pruned, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


class TrainData_deepCSV_gradient_pruned(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches([    'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig'],
                             6)
        
        self.addBranches([    'TagVarCSVTrk_trackDecayLenVal'],
                             5)

        self.addBranches([    'TagVarCSVTrk_trackDeltaR'],
                             2)

        self.addBranches([    'TagVarCSVTrk_trackPtRatio'],
                             1)

        
        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_deepCSV_gradient_pruned, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


class TrainData_deepCSV_smaller(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_vertexCategory',
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dSigAboveCharm', 
                           'TagVarCSV_jetNSelectedTracks', 'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal'],
                             5)
        
        self.addBranches(['TagVarCSVTrk_trackPtRatio'],
                             1)
        
        self.addBranches(['TagVarCSVTrk_trackDeltaR'],
                             2)
        
        self.addBranches(['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel'],
                             3)
        
        
        self.addBranches(['TagVarCSV_trackEtaRel'],2)

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
        super(TrainData_deepCSV_smaller, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


class TrainData_deepCSV_slim_more(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_vertexCategory',
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dSigAboveCharm', 
                           'TagVarCSV_jetNSelectedTracks', 'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig'],
                             2)
        
        self.addBranches(['TagVarCSVTrk_trackDecayLenVal'],
                             1)
        
        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexNTracks',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_deepCSV_slim_more, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]



class TrainData_deepCSV_slim_more_v2(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_vertexCategory',
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dSigAboveCharm', 
                           'TagVarCSV_jetNSelectedTracks', 'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig'],
                             3)
        
        self.addBranches(['TagVarCSVTrk_trackDecayLenVal',
                          'TagVarCSVTrk_trackPtRel'],
                             1)
        
        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexNTracks',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_deepCSV_slim_more_v2, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]




class TrainData_deepCSV_slim(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dSigAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig'],
                             5)

        self.addBranches(['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackDecayLenVal'],
                             3)
        
        
        self.addBranches(['TagVarCSV_trackEtaRel'],1)

        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        super(TrainData_deepCSV_slim, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]


class TrainData_deepCSV_rand(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel','randn_jet'])
       
        self.addBranches(['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal','randn_track'],
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
        super(TrainData_deepCSV_rand, self).readFromRootFile(filename, TupleMeanStd, weighter)
        ys = self.y[0]
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
        	raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
        mask = (flav_sum == 1) if self.remove else (np.ones(flav_sum.shape[0]) == 1)
        self.x = [self.x[0][mask]]
        self.y = [self.y[0][mask]]
        self.w = [self.w[0][mask]]

class TrainData_deepCSV_RNN(TrainData_fullTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''
    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_deepCSV_RNN, self).__init__()
        
        self.addBranches([
            'jet_pt', 'jet_eta',
            'TagVarCSV_jetNSecondaryVertices', 
            'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
            'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
            'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
            'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
            'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches([
            'TagVarCSVTrk_trackJetDistVal',
            'TagVarCSVTrk_trackPtRel', 
            'TagVarCSVTrk_trackDeltaR', 
            'TagVarCSVTrk_trackPtRatio', 
            'TagVarCSVTrk_trackSip3dSig', 
            'TagVarCSVTrk_trackSip2dSig', 
            'TagVarCSVTrk_trackDecayLenVal'
        ], 6)
        
        
        self.addBranches(['TagVarCSV_trackEtaRel'],4)

        self.addBranches([
            'TagVarCSV_vertexMass', 
            'TagVarCSV_vertexNTracks', 
            'TagVarCSV_vertexEnergyRatio',
            'TagVarCSV_vertexJetDeltaR',
            'TagVarCSV_flightDistance2dVal', 
            'TagVarCSV_flightDistance2dSig', 
            'TagVarCSV_flightDistance3dVal', 
            'TagVarCSV_flightDistance3dSig'
        ], 1)

        self.addBranches(['jet_corr_pt'])
        self.registerBranches(['gen_pt_WithNu'])
        self.regressiontargetclasses=['uncPt','Pt']


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        
        # split for convolutional network
        
        x_global = MeanNormZeroPad(
            filename,TupleMeanStd,
            [self.branches[0]],
            [self.branchcutoffs[0]],self.nsamples
        )
        
        x_cpf = MeanNormZeroPadParticles(
            filename,TupleMeanStd,
            self.branches[1],
            self.branchcutoffs[1],self.nsamples
        )
        
        x_etarel = MeanNormZeroPadParticles(
            filename,TupleMeanStd,
            self.branches[2],
            self.branchcutoffs[2],self.nsamples
        )
        
        x_sv = MeanNormZeroPadParticles(
            filename,TupleMeanStd,
            self.branches[3],
            self.branchcutoffs[3],self.nsamples
        )
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        npy_array = self.readTreeFromRootToTuple(filename)
        
        reg_truth=npy_array['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=npy_array['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
        for i in range(self.nsamples):
            correctionfactor[i]=reg_truth[i]/reco_pt[i]

        truthtuple =  npy_array[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        self.x=[x_global, x_cpf, x_etarel, x_sv, reco_pt]
        self.y=[alltruth,correctionfactor]
        #self._normalize_input_(weighter, npy_array)

        
    
    

class TrainData_deepCSV_RNN_Deeper(TrainData_deepCSV_RNN):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''
    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_deepCSV_RNN_Deeper, self).__init__()
        self.branchcutoffs = [1, 20, 13, 4, 1]

