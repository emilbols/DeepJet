
export DEEPJETCORE=../DeepJetCore

THISDIR=`pwd`
cd $DEEPJETCORE
source env.sh
cd $THISDIR
export PYTHONPATH=`pwd`/modules:$PYTHONPATH
export DEEPJET=`pwd`
#export LD_LIBRARY_PATH=/afs/cern.ch/user/j/jkiesele/eos_DeepJet/.lib:$LD_LIBRARY_PATH
