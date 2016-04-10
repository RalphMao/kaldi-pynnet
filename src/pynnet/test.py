import _nnet
import numpy as np
import IPython

net = _nnet.Nnet()
net.read('/home/maohz12/online_50h_Tsinghua/exp_train_50h/lstm_karel_bak/nnet/nnet_iter14_learnrate7.8125e-07_tr1.2687_cv1.6941')

# Test1
blobs = net.layers[0].get_params()
x = blobs[1].data.flatten()
x_test = np.fromfile('test/1.bin', 'f')
assert np.sum(abs(x-x_test)) < 1e-5
x = blobs[4].data.flatten()
x_test = np.fromfile('test/4.bin', 'f')
assert np.sum(abs(x-x_test)) < 1e-5

blobs[1].data[:] = np.arange(blobs[1].data.size).reshape(blobs[1].data.shape)
blobs[4].data[:] = np.arange(blobs[4].data.size).reshape(blobs[4].data.shape)

net.layers[0].set_params(blobs)
net.write('test/test_nnet', 0)

pointer, read_only_flag = blobs[1].data.__array_interface__['data']
# Test 2
data_copy = blobs[1].data.copy()
del net
pointer, read_only_flag = blobs[1].data.__array_interface__['data']
assert np.sum(abs(blobs[1].data - data_copy)) < 1e-5

# Test 3
net = _nnet.Nnet()
net.read('test/test_nnet')
blobs_new = net.layers[0].get_params()

x = blobs[1].data
x_test = blobs_new[1].data
assert np.sum(abs(x-x_test)) < 1e-5

x = blobs[4].data
x_test = blobs_new[4].data
assert np.sum(abs(x-x_test)) < 1e-5

print "Test passed"
