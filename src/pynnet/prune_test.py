
from pynnet.nnetrw import Nnet
import os
import pickle
import numpy as np

def get_LstmProjectedStreams(blobs):
    mats = []
    def split_blob(blob):
        assert blob.data.shape[0]%4 == 0
        stride = blob.data.shape[0] / 4
        return np.split(blob.data, np.arange(1,4)*stride)
    mats.extend(split_blob(blobs[0]))
    mats.extend(split_blob(blobs[1]))
    mats.append(blobs[6].data)
    names = ['w_g_x', 'w_i_x', 'w_f_x','w_o_x'] + ['w_g_r', 'w_i_r', 'w_f_r', 'w_o_r'] + ['w_r_m']
    return mats,names

def get_AffineTransform(blobs):
    return [blobs[0].data], ['w']

def get_function(name):
    return globals()['get_' + name]

def prune_mat(mat, sparsity):
    flatten_data = mat.flatten()
    rank = np.argsort(abs(flatten_data))
    flatten_data[rank[:-int(rank.size * sparsity)]] = 0
    flatten_data = flatten_data.reshape(mat.shape)
    np.copyto(mat, flatten_data)

def get_mats(net):
    mats = []
    names = []
    for layer in net.params:
        if len(net.params[layer]) == 0:
            continue
        func = get_function(net.layers[layer])
        mats_, names_ = func(net.params[layer])
        names_ = map(lambda x:net.layers[layer]+str(layer) + '_' +  x, names_)
        mats.extend(mats_)
        names.extend(names_)
    return mats, names

def test_net(net):
    net.Write('exp_train_50h/lstm_karel/final.nnet')
    flag = os.system('./test_wer.sh >>log 2>&1')
    assert flag == 0
    os.system('cat exp_train_50h/lstm_karel/decode_testset_test8000/wer_* | grep WER > results')

    lines = open('results').readlines()
    wer = map(lambda x:float(x.split()[1]), lines)
    os.system('rm results')
    return min(wer)

def main():
    net = Nnet('exp_train_50h/lstm_karel_bak/final.nnet.bak')
    sparsities = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75]

    all_mats,names = get_mats(net)
    all_mats_bak = map(lambda x:x.copy(), all_mats)

    wer = np.zeros((len(all_mats), len(sparsities)))
    f = open('results.csv', 'w')
    for id_t, (mat, mat_bak) in enumerate(zip(all_mats, all_mats_bak)):
        print "Pruning mat:", names[id_t]
        for j_t, sparsity in enumerate(sparsities):
            prune_mat(mat, sparsity)
            wer[id_t, j_t] = test_net(net)
            np.copyto(mat, mat_bak)
            print "Results:%s, %.2f, %.2f"%(names[id_t], sparsity, wer[id_t,j_t])

        f.write(("%s, " + '%.2f, '*len(sparsities) +"\n")%((names[id_t],) +  tuple(wer[id_t])))

    f.close()
    pickle.dump((names, wer), open('results.pkl','wb'))

if __name__ == "__main__":
    main()
