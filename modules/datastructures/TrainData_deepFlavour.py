#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.models import Model
import warnings
warnings.warn("DeepJet_models.py is deprecated and will be removed! Please move to the models directory", DeprecationWarning)

#from keras.layers.core import Reshape, Masking, Permute
from keras.layers import Reshape, Masking, Permute
#from keras.layers.pooling import MaxPooling2D
from keras.layers import MaxPool2D
#fix for dropout on gpus

#import tensorflow
#from tensorflow.python.ops import control_flow_ops 
#tensorflow.python.control_flow_ops = control_flow_ops

from TrainDataDeepJet import TrainData_fullTruth,TrainData_forTest,TrainData_simpleTruth
from TrainDataDeepJet import fileTimeOut,TrainData_QGOnly

class TrainData_testingClass(TrainData_forTest):
   

    def __init__(self):
        TrainData_forTest.__init__(self)
        self.addBranches(['x'])

    def readFromRootFile(self,filename,TupleMeanStd,weighter):


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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        weights=numpy.empty(self.nsamples)
        weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global]
        self.y=[alltruth]


class TrainData_deepFlavour_FT(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          #'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]



class TrainData_deepCSV_RNN_v2(TrainData_simpleTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_simpleTruth.__init__(self)

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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]




class TrainData_deepCSV_RNN_v2_moreTracks(TrainData_simpleTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_simpleTruth.__init__(self)

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
                         25)


        self.addBranches(['TagVarCSV_trackEtaRel'],4)
        
        self.addBranches(['TagVarCSV_vertexMass',
                          'TagVarCSV_vertexNTracks',
                          'TagVarCSV_vertexEnergyRatio',
                          'TagVarCSV_vertexJetDeltaR',
                          'TagVarCSV_flightDistance2dVal',
                          'TagVarCSV_flightDistance2dSig',
                          'TagVarCSV_flightDistance3dVal',
                          'TagVarCSV_flightDistance3dSig'],
                                 4)
        
        
       
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,None,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]



class TrainData_deepFlavour_FT_lrpatt3(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          'Cpfcan_drminsv',
                          #'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3dsig',
                          'sv_costhetasvpv'
                          ],
                          4)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]



class TrainData_deepFlavour_FT_lrpatt2(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]


class TrainData_deepFlavour_FT_lrpatt2_2(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]

class TrainData_deepFlavour_FT_lrpatt2_cuttracks(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw'
                              ],
                             21)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma'
                          ],
                         14)
        
        
        self.addBranches(['sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]

        
class TrainData_deepFlavour_FT_grad_skim(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          'Cpfcan_chi2'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        


class TrainData_deepFlavour_FT_lrp_skim(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          ])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal', 
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          ],
                          4)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        

class TrainData_deepFlavour_FT_skim(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv'],
                             18)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma'
                          ],
                         18)
        
        
        self.addBranches(['sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_dxysig',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio'
                          ],
                          2)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        

class TrainData_deepFlavour_FT_global_skim(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        

        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv'])
       
        self.addBranches(['Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          'Cpfcan_quality'
                              ],
                             18)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv'
                          ],
                         18)
        
        
        self.addBranches(['sv_deltaR',
                          'sv_costhetasvpv',
                          'sv_enratio'
                          ],
                          2)

       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]


class TrainData_deepFlavour(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          #'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
        
       
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        



class TrainData_deepFlavour_FT_reg(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          #'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)
        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_vertex_full(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          #'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)
        
        self.addBranches(['TagVarCSV_vertexMass',
                          'TagVarCSV_vertexNTracks',
                          'TagVarCSV_vertexEnergyRatio',
                          'TagVarCSV_vertexJetDeltaR',
                          'TagVarCSV_flightDistance2dVal',
                          'TagVarCSV_flightDistance2dSig',
                          'TagVarCSV_flightDistance3dVal',
                          'TagVarCSV_flightDistance3dSig'],
                                 4)
        
        
        self.addBranches(['seed_pt',
                          'seed_eta',
                          'seed_phi',
                          'seed_mass',
                          'seed_dz',
                          'seed_dxy',
                          'seed_3D_ip',
                          'seed_3D_sip',
                          'seed_2D_ip',
                          'seed_2D_sip',
                          'seed_3D_signedIp',
                          'seed_3D_signedSip',
                          'seed_2D_signedIp',
                          'seed_2D_signedSip',
                          'seed_3D_TrackProbability',
                          'seed_2D_TrackProbability',
                          'seed_chi2reduced',
                          'seed_nPixelHits',
                          'seed_nHits',
                          'seed_jetAxisDlength',
                          'seed_jetAxisDistance'
                          ],
                          10)

        self.addBranches(['nearTracks_pt',
                          'nearTracks_eta',
                          'nearTracks_phi',
                          'nearTracks_mass',
                          'nearTracks_dz',
                          'nearTracks_dxy',
                          'nearTracks_3D_ip',
                          'nearTracks_3D_sip',
                          'nearTracks_2D_ip',
                          'nearTracks_2D_sip',
                          'nearTracks_PCAonSeed_x',
                          'nearTracks_PCAonSeed_y',
                          'nearTracks_PCAonSeed_z',
                          'nearTracks_PCAonSeed_xerr',
                          'nearTracks_PCAonSeed_yerr',
                          'nearTracks_PCAonSeed_zerr',
                          'nearTracks_PCAonTrack_x',
                          'nearTracks_PCAonTrack_y',
                          'nearTracks_PCAonTrack_z',
                          'nearTracks_PCAonTrack_xerr',
                          'nearTracks_PCAonTrack_yerr',
                          'nearTracks_PCAonTrack_zerr',
                          'nearTracks_dotprodTrack',
                          'nearTracks_dotprodSeed',
                          'nearTracks_dotprodTrackSeed2D',
                          'nearTracks_dotprodTrackSeed3D',
                          'nearTracks_dotprodTrackSeedVectors2D',
                          'nearTracks_dotprodTrackSeedVectors3D',
                          'nearTracks_PCAonSeed_pvd',
                          'nearTracks_PCAonTrack_pvd',
                          'nearTracks_PCAjetAxis_dist',
                          'nearTracks_PCAjetMomenta_dotprod',
                          'nearTracks_PCAjetDirs_DEta',
                          'nearTracks_PCAjetDirs_DPhi'
                           ],
                          200)

        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_vtx = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                           self.branches[3],
                                           self.branchcutoffs[3],self.nsamples)

        x_seeds = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                           self.branches[4],
                                           self.branchcutoffs[4],self.nsamples)

        x_near = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[5],
                                          self.branchcutoffs[5],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            x_seeds=x_seeds[notremoves > 0]
            x_near=x_near[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_vtx,x_seeds,x_near,reco_pt]
        self.y=[alltruth,correctionfactor]



class TrainData_deepFlavour_FT_reg_vertex(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          #'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)
        
        self.addBranches(['TagVarCSV_vertexMass',
                          'TagVarCSV_vertexNTracks',
                          'TagVarCSV_vertexEnergyRatio',
                          'TagVarCSV_vertexJetDeltaR',
                          'TagVarCSV_flightDistance2dVal',
                          'TagVarCSV_flightDistance2dSig',
                          'TagVarCSV_flightDistance3dVal',
                          'TagVarCSV_flightDistance3dSig'],
                                 4)
        
        
        self.addBranches(['seed_pt',
                          'seed_eta',
                          'seed_phi',
                          'seed_mass',
                          'seed_dz',
                          'seed_dxy',
                          'seed_3D_ip',
                          'seed_3D_sip',
                          'seed_2D_ip',
                          'seed_2D_sip',
                          'seed_3D_signedIp',
                          'seed_3D_signedSip',
                          'seed_2D_signedIp',
                          'seed_2D_signedSip',
                          'seed_3D_TrackProbability',
                          'seed_jetAxisDlength',
                          'seed_jetAxisDistance'
                          ],
                          5)

        self.addBranches(['nearTracks_pt',
                          'nearTracks_eta',
                          'nearTracks_phi',
                          'nearTracks_mass',
                          'nearTracks_dz',
                          'nearTracks_dxy',
                          'nearTracks_3D_ip',
                          'nearTracks_3D_sip',
                          'nearTracks_2D_ip',
                          'nearTracks_2D_sip',
                          'nearTracks_PCAonSeed_x',
                          'nearTracks_PCAonSeed_y',
                          'nearTracks_PCAonSeed_z',
                          'nearTracks_PCAonSeed_xerr',
                          'nearTracks_PCAonSeed_yerr',
                          'nearTracks_PCAonSeed_zerr',
                          'nearTracks_PCAonTrack_x',
                          'nearTracks_PCAonTrack_y',
                          'nearTracks_PCAonTrack_z',
                          'nearTracks_PCAonTrack_xerr',
                          'nearTracks_PCAonTrack_yerr',
                          'nearTracks_PCAonTrack_zerr',
                          'nearTracks_dotprodSeed',
                          'nearTracks_dotprodTrack',
                          'nearTracks_PCAjetAxis_dist',
                          'nearTracks_PCAjetMomenta_dotprod'
                           ],
                          100)

        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_vtx = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                           self.branches[3],
                                           self.branchcutoffs[3],self.nsamples)

        x_seeds = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                           self.branches[4],
                                           self.branchcutoffs[4],self.nsamples)

        x_near = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[5],
                                          self.branchcutoffs[5],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            x_seeds=x_seeds[notremoves > 0]
            x_near=x_near[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_vtx,x_seeds,x_near,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_phi(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_phirel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv'
                          ],
                         25)
        
        
        self.addBranches(['sv_etarel',
                          'sv_phirel',
                          'sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)
        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_btv(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_ptrel', 
                              ],
                             25)
        
        self.addBranches(['sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_enratio',
                          ],
                          4)
        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_skim(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw'
                              ],
                             21)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma'
                          ],
                         14)
        
        
        self.addBranches(['sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_skim_noPuppi(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar'
                              ],
                             21)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma'
                          ],
                         14)
        
        
        self.addBranches(['sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]        



class TrainData_deepFlavour_FT_reg_noScale(TrainData_deepFlavour_FT_reg):
    
    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_reg.__init__(self)
        
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,None,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]




class TrainData_deepFlavour_FT_reg_vertex_noScale(TrainData_deepFlavour_FT_reg_vertex):
    
    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_reg_vertex.__init__(self)
        
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_vtx = MeanNormZeroPadParticles(filename,None,
                                         self.branches[3],
                                         self.branchcutoffs[3],self.nsamples)
        
        x_seeds = MeanNormZeroPadParticles(filename,None,
                                           self.branches[4],
                                           self.branchcutoffs[4],self.nsamples)
        
        x_near = MeanNormZeroPadParticles(filename,None,
                                          self.branches[5],
                                          self.branchcutoffs[5],self.nsamples)
        

        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            x_seeds=x_seeds[notremoves > 0]
            x_near=x_near[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_vtx,x_seeds,x_near,reco_pt]
        self.y=[alltruth,correctionfactor]



class TrainData_deepFlavour_FT_reg_vertex_full_noScale(TrainData_deepFlavour_FT_reg_vertex_full):
    
    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_reg_vertex_full.__init__(self)
        
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_vtx = MeanNormZeroPadParticles(filename,None,
                                         self.branches[3],
                                         self.branchcutoffs[3],self.nsamples)
        
        x_seeds = MeanNormZeroPadParticles(filename,None,
                                           self.branches[4],
                                           self.branchcutoffs[4],self.nsamples)
        
        x_near = MeanNormZeroPadParticles(filename,None,
                                          self.branches[5],
                                          self.branchcutoffs[5],self.nsamples)
        

        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            x_seeds=x_seeds[notremoves > 0]
            x_near=x_near[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_vtx,x_seeds,x_near,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_noScale_skim(TrainData_deepFlavour_FT_reg_skim):
    
    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_reg_skim.__init__(self)
        
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,None,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]




class TrainData_deepFlavour_FT_reg_noScale_btv(TrainData_deepFlavour_FT_reg_btv):
    
    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_reg_btv.__init__(self)
        
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,None,
                                        self.branches[2],
                                        self.branchcutoffs[2],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_noScale_skim_noPuppi(TrainData_deepFlavour_FT_reg_skim_noPuppi):
    
    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_reg_skim_noPuppi.__init__(self)
        
        
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,None,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_noScale_phi(TrainData_deepFlavour_FT_reg_phi):
    
    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_reg_phi.__init__(self)

        
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,None,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]



class TrainData_deepFlavour_QGOnly_reg(TrainData_QGOnly):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_QGOnly.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)
        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        x_reg = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[4]],
                                   [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            x_reg=x_reg[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,x_reg]
        self.y=[alltruth,correctionfactor]
        


class TrainData_deepFlavour_FT_map(TrainData_deepFlavour_FT):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT.__init__(self)
        
        
        self.registerBranches(['Cpfcan_ptrel','Cpfcan_eta','Cpfcan_phi',
                               'Npfcan_ptrel','Npfcan_eta','Npfcan_phi',
                               'nCpfcand','nNpfcand',
                               'jet_eta','jet_phi','jet_pt'])
        

        
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply,createCountMap,createDensity, MeanNormZeroPad, createDensityMap, MeanNormZeroPadParticles
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
        
        
        
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        #here the difference starts
        nbins=8
        
        x_chmap = createDensity (filename,
                              inbranches=['Cpfcan_ptrel',
                                          'Cpfcan_etarel',
                                          'Cpfcan_phirel'], 
                              modes=['sum',
                                     'average',
                                     'average'],
                              nevents=self.nsamples,
                              dimension1=['Cpfcan_eta','jet_eta',nbins,0.45], 
                              dimension2=['Cpfcan_phi','jet_phi',nbins,0.45],
                              counterbranch='nCpfcand',
                              offsets=[-1,-0.5,-0.5])
        
        x_neumap = createDensity (filename,
                              inbranches=['Npfcan_ptrel',
                                          'Npfcan_etarel',
                                          'Npfcan_phirel'], 
                              modes=['sum',
                                     'average',
                                     'average'],
                              nevents=self.nsamples,
                              dimension1=['Npfcan_eta','jet_eta',nbins,0.45], 
                              dimension2=['Npfcan_phi','jet_phi',nbins,0.45],
                              counterbranch='nCpfcand',
                              offsets=[-1,-0.5,-0.5])
        
        
        x_chcount = createCountMap(filename,TupleMeanStd,
                                   self.nsamples,
                                   ['Cpfcan_eta','jet_eta',nbins,0.45],
                                   ['Cpfcan_phi','jet_phi',nbins,0.45],
                                   'nCpfcand')                  
                                                                
        x_neucount = createCountMap(filename,TupleMeanStd,      
                                   self.nsamples,               
                                   ['Npfcan_eta','jet_eta',nbins,0.45],
                                   ['Npfcan_phi','jet_phi',nbins,0.45],
                                   'nNpfcand')
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            
            x_chmap=x_chmap[notremoves > 0]
            x_neumap=x_neumap[notremoves > 0]
            
            x_chcount=x_chcount[notremoves > 0]
            x_neucount=x_neucount[notremoves > 0]
            
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        

        x_map = numpy.concatenate((x_chmap,x_neumap,x_chcount,x_neucount), axis=3)
        
        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,x_map]
        self.y=[alltruth]
        
        
        

class TrainData_deepFlavour_FT_map_reg(TrainData_deepFlavour_FT_map):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_map.__init__(self)
        
        self.regressiontargetclasses=['uncPt','Pt']

        self.registerBranches(['jet_corr_pt'])
        
        
        
        self.registerBranches(['jet_corr_pt','gen_pt_WithNu'])
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply,createCountMap,createDensity, MeanNormZeroPad, createDensityMap, MeanNormZeroPadParticles
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
        
        
        
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        #here the difference starts
        nbins=8
        
        x_chmap = createDensity (filename,
                              inbranches=['Cpfcan_ptrel',
                                          'Cpfcan_etarel',
                                          'Cpfcan_phirel'], 
                              modes=['sum',
                                     'average',
                                     'average'],
                              nevents=self.nsamples,
                              dimension1=['Cpfcan_eta','jet_eta',nbins,0.45], 
                              dimension2=['Cpfcan_phi','jet_phi',nbins,0.45],
                              counterbranch='nCpfcand',
                              offsets=[-1,-0.5,-0.5])
        
        x_neumap = createDensity (filename,
                              inbranches=['Npfcan_ptrel',
                                          'Npfcan_etarel',
                                          'Npfcan_phirel'], 
                              modes=['sum',
                                     'average',
                                     'average'],
                              nevents=self.nsamples,
                              dimension1=['Npfcan_eta','jet_eta',nbins,0.45], 
                              dimension2=['Npfcan_phi','jet_phi',nbins,0.45],
                              counterbranch='nCpfcand',
                              offsets=[-1,-0.5,-0.5])
        
        
        x_chcount = createCountMap(filename,TupleMeanStd,
                                   self.nsamples,
                                   ['Cpfcan_eta','jet_eta',nbins,0.45],
                                   ['Cpfcan_phi','jet_phi',nbins,0.45],
                                   'nCpfcand')                  
                                                                
        x_neucount = createCountMap(filename,TupleMeanStd,      
                                   self.nsamples,               
                                   ['Npfcan_eta','jet_eta',nbins,0.45],
                                   ['Npfcan_phi','jet_phi',nbins,0.45],
                                   'nNpfcand')
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        regtruth=Tuple['gen_pt_WithNu']
        regreco=Tuple['jet_corr_pt']
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            
            x_chmap=x_chmap[notremoves > 0]
            x_neumap=x_neumap[notremoves > 0]
            
            x_chcount=x_chcount[notremoves > 0]
            x_neucount=x_neucount[notremoves > 0]
            
            alltruth=alltruth[notremoves > 0]
            
            regreco=regreco[notremoves > 0]
            regtruth=regtruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        

        x_map = numpy.concatenate((x_chmap,x_neumap,x_chcount,x_neucount), axis=3)
        
        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,x_map,regreco]
        self.y=[alltruth,regtruth]        

        
class TrainData_image(TrainData_fullTruth):
    '''
    This class is for simple jetimiging
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_image,self).__init__()
        self.regressiontargetclasses=['uncPt','Pt']

        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','rho'])
        self.registerBranches(['Cpfcan_ptrel','Cpfcan_eta','Cpfcan_phi',
                               'Npfcan_ptrel','Npfcan_eta','Npfcan_phi',
                               'nCpfcand','nNpfcand',
                               'jet_eta','jet_phi','jet_pt'])

        self.regtruth='gen_pt_WithNu'
        self.regreco='jet_corr_pt'
        
        self.registerBranches([self.regtruth,self.regreco])
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, createDensityMap,createCountMap, MeanNormZeroPadParticles
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)    
        
        #here the difference starts
        x_chmap = createDensityMap(filename,TupleMeanStd,
                                   'Cpfcan_ptrel',
                                   self.nsamples,
                                   ['Cpfcan_eta','jet_eta',20,0.5],
                                   ['Cpfcan_phi','jet_phi',20,0.5],
                                   'nCpfcand',-1, weightbranch='Cpfcan_puppiw')
        
        x_chcount = createCountMap(filename,TupleMeanStd,
                                   self.nsamples,
                                   ['Cpfcan_eta','jet_eta',20,0.5],
                                   ['Cpfcan_phi','jet_phi',20,0.5],
                                   'nCpfcand')
        
        x_neumap = createDensityMap(filename,TupleMeanStd,
                                   'Npfcan_ptrel',
                                   self.nsamples,
                                   ['Npfcan_eta','jet_eta',20,0.5],
                                   ['Npfcan_phi','jet_phi',20,0.5],
                                   'nNpfcand',-1, weightbranch='Npfcan_puppiw')
        
        x_neucount = createCountMap(filename,TupleMeanStd,
                                   self.nsamples,
                                   ['Npfcan_eta','jet_eta',20,0.5],
                                   ['Npfcan_phi','jet_phi',20,0.5],
                                   'nNpfcand')
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
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
            weights=numpy.ones(self.nsamples)

        pttruth=Tuple[self.regtruth]
        ptreco=Tuple[self.regreco]         
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        x_map = numpy.concatenate((x_chmap,x_chcount,x_neumap,x_neucount), axis=3)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_map=x_map[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            pttruth=pttruth[notremoves > 0]
            ptreco=ptreco[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_map,ptreco]
        self.y=[alltruth,pttruth]

    @staticmethod
    def base_model(input_shapes):
        from keras.layers import Input
        from keras.layers.core import Masking
        x_global  = Input(shape=input_shapes[0])
        x_map = Input(shape=input_shapes[1])
        x_ptreco  = Input(shape=input_shapes[2])

        x =   Convolution2D(64, (8,8)  , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x_map)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x =   Convolution2D(64, (4,4) , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x =   Convolution2D(64, (4,4)  , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        #x = merge( [x, x_global] , mode='concat')
        # linear activation for regression and softmax for classification
        x = Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)
        #x = merge([x, x_ptreco], mode='concat')
        return [x_global, x_map, x_ptreco], x

    @staticmethod
    def regression_generator(generator):
        for X, Y in generator:
            yield X, Y[1]#.astype(int)

    @staticmethod
    def regression_model(input_shapes):
        inputs, x = TrainData_image.base_model(input_shapes)
        predictions = Dense(2, activation='linear',init='normal')(x)
        return Model(inputs=inputs, outputs=predictions)

    @staticmethod
    def classification_generator(generator):
        for X, Y in generator:
            yield X, Y[0]#.astype(int)

    @staticmethod
    def classification_model(input_shapes, nclasses):
        inputs, x = TrainData_image.base_model(input_shapes)
        predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
        return Model(inputs=inputs, outputs=predictions)

    @staticmethod
    def model(input_shapes, nclasses):
        inputs, x = TrainData_image.base_model(input_shapes)
        predictions = [
            Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x),
            Dense(2, activation='linear',init='normal')(x)
        ]
        return Model(inputs=inputs, outputs=predictions)
        


class TrainData_deepFlavour_cleaninput(TrainData_deepFlavour_FT_reg_noScale):
    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_deepFlavour_cleaninput, self).__init__()
        self.branches[1].remove('Cpfcan_quality')
        self.branches[1].append('Cpfcan_lostInnerHits')#Cpfcan_numberOfPixelHits
        
class TrainData_deepFlavour_cleanBTVOnly(TrainData_fullTruth):
    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_deepFlavour_cleanBTVOnly, self).__init__()
        ##  ---- GLOBAL ----
        ##  'jetPt': False, #$
        ##  'jetAbsEta': False, #$ 
        ##  'jetNSecondaryVertices': False, #$
        ##  'jetNSelectedTracks': False, #$
        ##  'jetNTracksEtaRel': False, #$
        ##  'vertexCategory': False, #$
        ##  'trackSip3dValAboveCharm': False, #$
        ##  'trackSip2dSigAboveCharm': False, #$
        ##  'trackSip2dValAboveCharm': False, #$
        ##  'trackSip3dSigAboveCharm': False  #$

        ##  'trackSumJetEtRatio': False, #$
        ##  'trackSumJetDeltaR': False, #$

        ##  ---- VERTEX ----
        ##  'vertexJetDeltaR': False, #$
        ##  'vertexMass': False, #$
        ##  'vertexNTracks': False, #$ 
        ##  'vertexEnergyRatio': False, #$
        ##  'flightDistance3dSig': False, #$
        ##  'flightDistance3dVal': False, #$
        ##  'flightDistance2dVal': False, #$
        ##  'flightDistance2dSig': False, #$

        ##  ---- TRACKS ----
        ##  'trackEtaRel': True, #$
        ##  'trackDecayLenVal': True, 
        ##  'trackJetDist': True, #$
        ##  'trackPtRatio': True, #$
        ##  'trackDeltaR': True, #$
        ##  'trackSip2dSig': True, #$
        ##  'trackPtRel': True, #$
        ##  'trackSip3dSig': True, #$
        self.addBranches([
                'jet_pt', 'jet_eta', #$
                'TagVarCSV_jetNSecondaryVertices', #$
                'TagVarCSV_trackSumJetEtRatio', #$ 
                'TagVarCSV_trackSumJetDeltaR', 
                'TagVarCSV_vertexCategory', #$
                'TagVarCSV_trackSip2dValAboveCharm', #$
                'TagVarCSV_trackSip2dSigAboveCharm', #$
                'TagVarCSV_trackSip3dValAboveCharm', #$
                'TagVarCSV_trackSip3dSigAboveCharm', #$
                'TagVarCSV_jetNSelectedTracks', #$
                'TagVarCSV_jetNTracksEtaRel' #$
                ])
       
        self.addBranches([
                'Cpfcan_BtagPf_trackEtaRel', #$
                'Cpfcan_BtagPf_trackPtRel', #$
                'Cpfcan_BtagPf_trackDeltaR', #$
                'Cpfcan_BtagPf_trackSip2dSig', #$
                'Cpfcan_BtagPf_trackSip3dSig', #$
                'Cpfcan_BtagPf_trackJetDistVal', #$
                'Cpfcan_ptrel', #$
                #'Cpfcan_BtagPf_trackJetDistSig', #?                          
                #trackDecayLenVal #?
                ], 20)
        
        self.addBranches([
                'sv_deltaR', #$
                'sv_mass', #$
                'sv_ntracks', #$
                'sv_dxy',  #$
                'sv_dxysig', #$
                'sv_d3d', #$
                'sv_d3dsig', #$
                'sv_enratio', #$
                ], 4)
        
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
            filename,None,
            [self.branches[0]],
            [self.branchcutoffs[0]],self.nsamples
        )
        
        x_cpf = MeanNormZeroPadParticles(
            filename,None,
            self.branches[1],
            self.branchcutoffs[1],self.nsamples
        )
                
        x_sv = MeanNormZeroPadParticles(
            filename,None,
            self.branches[2],
            self.branchcutoffs[2],self.nsamples
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
        
        self.x=[x_global, x_cpf, x_sv, reco_pt]
        self.y=[alltruth,correctionfactor]
        self._normalize_input_(weighter, npy_array)


class TrainData_deepFlavour_nopuppi(TrainData_deepFlavour_FT_reg_noScale):
	def __init__(self):
		'''
		Constructor FIXME
		'''
		super(TrainData_deepFlavour_nopuppi, self).__init__()
		self.branches[1].remove('Cpfcan_puppiw')
                self.branches[1].remove('Cpfcan_VTX_ass')
		self.branches[2].remove('Npfcan_puppiw')



class TrainData_deepFlavour_FT_reg_more_tracks(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             40)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv'
                          ],
                         30)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          6)
        
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]


class TrainData_deepFlavour_FT_reg_more_tracks_noScale(TrainData_deepFlavour_FT_reg_more_tracks):
    
    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepFlavour_FT_reg_more_tracks.__init__(self)
        
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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,None,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        #x_reg = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[4]],
        #                           [self.branchcutoffs[4]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        reg_truth=Tuple['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=Tuple['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
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
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
            reco_pt=reco_pt[notremoves > 0]
            correctionfactor=correctionfactor[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights,weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,reco_pt]
        self.y=[alltruth,correctionfactor]
