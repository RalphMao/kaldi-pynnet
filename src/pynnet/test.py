import _nnet

net = _nnet.Nnet()
net.read('/home/maohz12/online_50h_Tsinghua/exp_train_50h/lstm_karel_bak/nnet/nnet_iter14_learnrate7.8125e-07_tr1.2687_cv1.6941')

blobs = net.layers[0].get_params()
x = blobs[1].data[4095].tolist()
x = blobs[4].data[-1].tolist()
import IPython
IPython.embed()
