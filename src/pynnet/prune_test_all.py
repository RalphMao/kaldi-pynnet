from prune_test import *

def get_prune_sparsities(mats, output_number = 10):
    sparsities = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75] + [1]
    sparsities = np.array(sparsities)
    mat_sizes = np.array(map(lambda x:x.size, mats))
    names, wer = pickle.load(open('results.pkl'))
    wer_all = np.zeros((wer.shape[0], wer.shape[1]+1))

    wer_all[:,:-1] = wer
    wer_all[:,-1] = 48.4

    wer_deri_avg = (wer_all[:,:-1] - wer_all[:,-1:]) / (sparsities[-1] - sparsities[:-1])
    wer_deri_weighted = wer_deri_avg / mat_sizes[:,None]

    med_wer_deri = np.median(wer_deri_weighted)

    threshs = np.arange(1,1+output_number).astype('f') / output_number * 2 * med_wer_deri
    import IPython
    for thresh in threshs:
        ratio = []
        error_inc = 0
        for mat_id in range(len(mat_sizes)):
            ids = np.where(wer_deri_weighted[mat_id] < thresh)[0]
            i = np.min(ids) if len(ids) > 0 else len(sparsities)-1
            ratio.append(sparsities[i])

            error_inc += wer_all[mat_id,i] - wer_all[mat_id, -1]

        print "======================================="
        print ("%.2f, "*len(ratio))%tuple(ratio)
        print "Error increased(estimated): ", error_inc

def test():
    net = Nnet('exp_train_50h/lstm_karel_bak/final.nnet.bak')
    all_mats,names = get_mats(net)
    get_prune_sparsities(all_mats)

def do_prune():
    mat_sparsities = [[1.00, 0.75, 0.25, 0.75, 0.40, 0.10, 0.25, 0.25, 1.00, 0.75, 0.50, 0.20, 0.25, 0.50, 1.00, 0.10, 0.10, 0.25, 0.20],
        [1.00, 0.75, 0.20, 0.50, 0.40, 0.10, 0.20, 0.25, 1.00, 0.75, 0.50, 0.20, 0.25, 0.50, 1.00, 0.10, 0.10, 0.25, 0.20]
        ]

    for sparsities in mat_sparsities:
        net = Nnet('exp_train_50h/lstm_karel_bak/final.nnet.bak')
        all_mats,names = get_mats(net)

        for mat, sparsity in zip(all_mats, sparsities):
            prune_mat(mat, sparsity)
        wer = test_net(net)
        print "Results:", wer

if __name__ == "__main__":
    do_prune()
