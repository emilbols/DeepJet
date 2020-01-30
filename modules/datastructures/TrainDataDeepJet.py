

from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy

def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)



class TrainData_forTest(TrainData):
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.truthclasses=['class1','class2']

        self.treename="deepntuplizer/tree"
        self.referenceclass='class1'
        
        
        self.registerBranches(self.truthclasses)
        self.registerBranches(['x'])
        
        self.weightbranchX='x'
        self.weightbranchY='x'
        
        self.weight_binX = numpy.array([-1,0.9,2.0],dtype=float)
        
        self.weight_binY = numpy.array(
            [-1,0.9,2.0],
            dtype=float
            )

        
             
        def reduceTruth(self, tuple_in):
        
            self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
            if tuple_in is not None:
                class1 = tuple_in['class1'].view(numpy.ndarray)
            
                class2 = tuple_in['class2'].view(numpy.ndarray)
                
                return numpy.vstack((class1,class2)).transpose()    
  
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd):
        from DeepJetCore.stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        #print('took ', sw.getAndReset(), ' seconds for getting tree entries')
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        truthtuple = Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
       
        newnsamp=x_all.shape[0]
        self.nsamples = newnsamp
        
        
        return x_all,alltruth


class TrainDataDeepJet(TrainData):
    '''
    Base class for DeepJet.
    TO create own b-tagging trainings, please inherit from this class
    '''
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.undefTruth=['isUndefined']
        self.referenceclass='isB'
        self.truthclasses=['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isCC',
                           'isGCC','isUD','isS','isG','isUndefined']
        
        
        #standard branches
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['jet_pt','jet_eta'])
        
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        
        self.weight_binX = numpy.array([10,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )
        
        
             
        self.reduceTruth(None)
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        #print('took ', sw.getAndReset(), ' seconds for getting tree entries')
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        #print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        #print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth, notremoves


class TrainDataDeepJet_noNorm(TrainData):
    '''
    Base class for DeepJet.
    TO create own b-tagging trainings, please inherit from this class
    '''
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.undefTruth=['isUndefined']
        self.referenceclass='isB'
        self.truthclasses=['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isCC',
                           'isGCC','isUD','isS','isG','isUndefined']
        
        
        #standard branches
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['jet_pt','jet_eta'])
        
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        
        self.weight_binX = numpy.array([10,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )
        
        
             
        self.reduceTruth(None)
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        #print('took ', sw.getAndReset(), ' seconds for getting tree entries')
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,None,self.branches,self.branchcutoffs,self.nsamples)
        
        #print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        #print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth, notremoves



class TrainDataDeepJet_btagana(TrainData):
    '''
    Base class for DeepJet.
    TO create own b-tagging trainings, please inherit from this class
    '''
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.referenceclass='isDATA'
        self.truthclasses=['isMC','isDATA']
        
        
        #standard branches
        self.registerBranches(self.truthclasses)
        self.registerBranches(['Jet_pt','n_PV'])
        
        self.weightbranchX='Jet_pt'
        self.weightbranchY='n_PV'
        
        self.weight_binX = numpy.array([0,
                50,55,60,65,75,80,85,90,95,100,105,110,115,120,125,
                130,135,140,145,150,155,160.,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [0,2.,5.,7.,10.,13.,15.,17.,20.,23,25.,27.,30.,33.,35.,37.,40.,43.,45.,50,55,60,65,70,75,500.],
            dtype=float
            )
        
        
             
        self.reduceTruth(None)
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        #print('took ', sw.getAndReset(), ' seconds for getting tree entries')
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        #print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        #print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth, notremoves


        
from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad

class TrainData_Flavour(TrainDataDeepJet):
    '''
    
    '''
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        weights,x_all,alltruth, _ =self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]

class TrainData_Flavour_noNorm(TrainDataDeepJet_noNorm):
    '''
    
    '''
    def __init__(self):
        TrainDataDeepJet_noNorm.__init__(self)
        self.clear()
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        weights,x_all,alltruth, _ =self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]

        
class TrainData_Flavour_MCDATA(TrainDataDeepJet_btagana):
    '''
    
    '''
    def __init__(self):
        TrainDataDeepJet_btagana.__init__(self)
        self.clear()
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        weights,x_all,alltruth, _ =self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]
        
     
     
class TrainData_simpleTruth(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isC','isUDSG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            allb = b+bl+blc

            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)            
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            g = tuple_in['isG'].view(numpy.ndarray)
            l = g + uds

            return numpy.vstack((allb,bb+gbb,c+cc+gcc,l)).transpose()
    
class TrainData_simpleTruth_MCDATA(TrainDataDeepJet_btagana):
    def __init__(self):
        TrainDataDeepJet_btagana.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isMC','isDATA']
        if tuple_in is not None:
            b = tuple_in['isMC'].view(numpy.ndarray)
            allb = b
            s = tuple_in['isDATA'].view(numpy.ndarray)
            l=s
            
            return numpy.vstack((allb,l)).transpose()
    
    
#    
#
#  DeepJet default classes
#
#

    
class TrainData_leptTruth(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDSG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)
            
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            lepb=bl+blc
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            l = g + uds
            
            return numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,l)).transpose()  
        
        
        

class TrainData_fullTruth(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            
            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)
            
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            lepb=bl+blc
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            
            
            return numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()    
  

class TrainData_QGOnly(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        self.undefTruth=['isUndefined']
        
        self.referenceclass='isUD'
        
        
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isUDS','isG']
        if tuple_in is not None:
            #b = tuple_in['isB'].view(numpy.ndarray)
            #bb = tuple_in['isBB'].view(numpy.ndarray)
            #gbb = tuple_in['isGBB'].view(numpy.ndarray)
            #
            #
            #bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            #blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            #lepb=bl+blc
            #
            #c = tuple_in['isC'].view(numpy.ndarray)
            #cc = tuple_in['isCC'].view(numpy.ndarray)
            #gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            
            
            return numpy.vstack((uds,g)).transpose()    

class TrainData_quarkGluon(TrainDataDeepJet):
    def __init__(self):
        super(TrainData_quarkGluon, self).__init__()
        self.referenceclass = 'isG'
        self.reducedtruthclasses=['isQ', 'isG']
        self.clear()
        
    def reduceTruth(self, tuple_in):
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            #bb = tuple_in['isBB'].view(numpy.ndarray) #this should be gluon?
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            c = tuple_in['isC'].view(numpy.ndarray)
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            q = ud+s#+c+blc+bl+b
            
            g = tuple_in['isG'].view(numpy.ndarray)
            return numpy.vstack((q, g)).transpose()    
        else:
            print('I got an empty tuple?')
        
        
