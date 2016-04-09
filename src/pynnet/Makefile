
CPLUS_INCLUDE_PATH=/home/maohz12/Boost.NumPy:/usr/include/python2.7
LIBRARY_PATH=/home/maohz12/Boost.NumPy/build/lib:/usr/lib/x86_64-linux-gnu
LD_LIBRARY_PATH=$(LIBRARY_PATH)
KALDI_SRC=/home/maohz12/kaldi-trunk/src

PYNNET=$(KALDI_SRC)/pynnet/_nnet.cc


include ../kaldi.mk
flag=$(PYNNET) $(LD_FLAGS) $(LDLIBS) -L$(KALDI_SRC)/lib -lkaldi-nnet -lkaldi-cudamatrix -lkaldi-matrix -lkaldi-base -lkaldi-util -lboost_python -lboost_numpy -lpython2.7

all:
	g++ -g -rdynamic --shared -o _nnet.so $(CXXFLAGS) $(flag)
