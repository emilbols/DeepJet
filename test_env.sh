
export DEEPJETCORE=../DeepJetCore

THISDIR=`pwd`
cd $DEEPJETCORE
source test_env.sh
cd $THISDIR
export PYTHONPATH=`pwd`/modules:$PYTHONPATH
export DEEPJET=`pwd`
export LD_LIBRARY_PATH=/afs/cern.ch/user/j/jkiesele/eos_DeepJet/.lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/afs/cern.ch/work/e/ebols/miniconda3/lib/libstdc++.so.6.0.24
