
CPLUS_INCLUDE_PATH=/home/maohz12/Boost.NumPy:/usr/include/python2.7
LIBRARY_PATH=/home/maohz12/Boost.NumPy/build/lib:/usr/lib/x86_64-linux-gnu
LD_LIBRARY_PATH=$(LIBRARY_PATH)
KALDI_SRC=/home/maohz12/kaldi-trunk/src

PYNNET=$(KALDI_SRC)/pynnet/_nnet.cc


include ../kaldi.mk
flag=$(PYNNET) $(LD_FLAGS) $(LDLIBS) -lboost_python -lboost_numpy

all:
	g++ --shared -o _nnet.so $(CXXFLAGS) $(flag)
