
import IPython
import os,sys
from collections import OrderedDict
import re
import numpy as np

source_ = r'''
{ Component::kAffineTransform,"<AffineTransform>" },
{ Component::kLinearTransform,"<LinearTransform>" },
{ Component::kConvolutionalComponent,"<ConvolutionalComponent>"},
{ Component::kConvolutional2DComponent,"<Convolutional2DComponent>"},
{ Component::kLstmProjectedStreams,"<LstmProjectedStreams>"},
{ Component::kBLstmProjectedStreams,"<BLstmProjectedStreams>"},
{ Component::kSoftmax,"<Softmax>" },
{ Component::kBlockSoftmax,"<BlockSoftmax>" },
{ Component::kSigmoid,"<Sigmoid>" },
{ Component::kTanh,"<Tanh>" },
{ Component::kDropout,"<Dropout>" },
{ Component::kLengthNormComponent,"<LengthNormComponent>" },
{ Component::kRbm,"<Rbm>" },
{ Component::kSplice,"<Splice>" },
{ Component::kCopy,"<Copy>" },
{ Component::kAddShift,"<AddShift>" },
{ Component::kRescale,"<Rescale>" },
{ Component::kKlHmm,"<KlHmm>" },
{ Component::kAveragePoolingComponent,"<AveragePoolingComponent>"},
{ Component::kAveragePooling2DComponent,"<AveragePooling2DComponent>"},
{ Component::kMaxPoolingComponent, "<MaxPoolingComponent>"},
{ Component::kMaxPooling2DComponent, "<MaxPooling2DComponent>"},
{ Component::kSentenceAveragingComponent,"<SentenceAveragingComponent>"},
{ Component::kSimpleSentenceAveragingComponent,"<SimpleSentenceAveragingComponent>"},
{ Component::kFramePoolingComponent, "<FramePoolingComponent>"},
{ Component::kParallelComponent, "<ParallelComponent>"},
'''

def get_first_key(string):
    x = re.search(r'<[a-zA-Z]+>',string)
    if x is not None:
        return string[x.start()+1:x.end()-1]
    else:
        return None

def get_keys(string):
    lines = string.split('\n')
    keys = set()
    for line in lines:
        keys.add(get_first_key(line))
    return keys

Supported_Layers = get_keys(source_)
if None in Supported_Layers:
    Supported_Layers.remove(None)

class Blob(object):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

class Nnet(object):
    def __init__(self):
        self.params = OrderedDict() #parameters of updatable layers
        self.layers = OrderedDict() #type of layers
        self.misc = OrderedDict() #type of layers

    def append_layer(self, type_t):
        assert type_t in Supported_Layers
        self.params[len(self.layers)] = []
        self.misc[len(self.layers)] = []
        self.layers[len(self.layers)] = type_t

    def append_params(self, params):
        self.params[len(self.layers)-1].append(Blob(params))

    def append_misc(self, string):
        if len(self.layers) == 0:
            return
        self.misc[len(self.layers)-1].append(string)

def ReadMat(f, line):
    if line.strip() == '':
        line = f.readline()
        assert line != ''
    flag = True
    data = []
    while flag:
        if ']' in line:
            flag = False
            assert line[line.index(']')+1:].strip() == ''
            line = line[:line.index(']')]
        if line.strip() != '':
            data.append(map(float, line.split()))
        if flag:
            line = f.readline()

    data = np.array(data)
    if data.dtype != np.dtype('float64'):
        IPython.embed()
    return np.squeeze(data.astype('f'))

def WriteMat(f, mat):
    if mat.ndim == 1:
        mat = mat[None,:]
    assert mat.ndim == 2
    f.write(' [\n')
    for i in range(mat.shape[0]):
        f.write(' '.join(map(str, mat[i])) + '\n')
    f.write(' ]\n')

def ReadNet(filename):
    #flag = os.system('nnet-copy --binary=false %s .tmp'%(filename))
    #assert flag == 0

    net = Nnet()
    f = open('.tmp','r')
    while True:
        line = f.readline()
        if line == '':
            break
        key = get_first_key(line)
        if key == '/Nnet':
            break
        elif key in Supported_Layers:
            net.append_layer(key)
            print "Get a Layer:" + key
            net.append_misc(line)
            assert '[' not in line
        elif '[' in line:
            idx = line.index('[')
            if line[:idx].strip() != '':
                net.append_misc(line[:idx])
            params = ReadMat(f, line[idx+1:])
            net.append_params(params)
        else:
            net.append_misc(line)
        print key

    #os.system('rm .tmp')
    return net

def WriteNet(net, filename):
    f = open(filename, 'w')

    f.write('<Nnet>\n')
    for layer in net.layers:
        f.writelines(net.misc[layer])
        for mat in net.params[layer]:
            WriteMat(f,mat.data)
    f.write('<Nnet>\n')
    f.close()

if __name__ == "__main__":
    
    net = ReadNet('/home/maohz12/online_50h_Tsinghua/exp_train_50h/lstm_karel_bak/final.nnet')
    WriteNet(net, 'test.nnet')
    net2 = ReadNet('test.nnet')
    IPython.embed()
