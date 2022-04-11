import sys
import torch
import logging
from dump import Dump
from base import Base

from pytorch2caffe.pnnx import Pnnx
from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer

from pytorch2caffe.op.relu import ReLU 
from pytorch2caffe.op.input import Input 
from pytorch2caffe.op.slice import Slice
from pytorch2caffe.op.swish import Swish
from pytorch2caffe.op.concat import Concat
from pytorch2caffe.op.linear import Linear
from pytorch2caffe.op.output import Output
from pytorch2caffe.op.pooling import Pooling
from pytorch2caffe.op.flatten import Flatten
from pytorch2caffe.op.dropout import Dropout
from pytorch2caffe.op.sigmoid import Sigmoid
from pytorch2caffe.op.conv import Convolution
from pytorch2caffe.op.batchnorm import BatchNorm
from pytorch2caffe.op.expression import Expression
from pytorch2caffe.op.adaptavgpooling import AdaptiveAvgPooling

from pytorch2caffe.op.debug import Debug




logger = logging.getLogger('Pytorch2Caffe')


OpMap = {
    'nn.ReLU': ReLU,
    'aten::relu_': ReLU,
    'torch.cat': Concat,
    'nn.Linear': Linear,
    'pnnx.Input': Input,
    'pnnx.Output': Output,
    'nn.Dropout': Dropout,
    'nn.Hardswish': Swish,
    'Tensor.slice': Slice,
    'nn.AvgPool2d': Pooling,
    'nn.MaxPool2d': Pooling,
    'nn.Conv2d': Convolution,
    'F.hardsigmoid': Sigmoid,
    'torch.flatten': Flatten,
    'nn.BatchNorm2d': BatchNorm,
    'pnnx.Expression': Expression,
    'nn.AdaptiveAvgPool2d': AdaptiveAvgPooling,
    'F.adaptive_avg_pool2d': AdaptiveAvgPooling,

#'nn.ChannelShuffle': Debug,
}


class Model(Base):

    def __init__(self, pytorch_file, param, inputs_tensor):
        super().__init__(None, None)
        self.pytorch_file = pytorch_file
        self.input_tensor = inputs_tensor
        self.version = ''

        input_tensor = torch.as_tensor(inputs_tensor)
        self.inputs_buf = [input_tensor]

        self.inputs = []
        self.inputs_shape = [list(input_tensor.shape)]

        self.param = param
        self.operators = []
        self.layers = []
        self.legacys = []
        self.setInited()


    def parse(self):
        logger.debug('Parsing the Pytorch Model...')

        self.pnnx = Pnnx(self.pytorch_file, self.inputs_shape)

        ops_types = self.pnnx.get_ops_type()
        for index, op_type in enumerate(ops_types):
            if op_type not in OpMap:
                errorMsg = 'Error: Operator [' + op_type + '] does not Support.\n'
                sys.exit(errorMsg)

            op = OpMap[op_type](self, self.pnnx, op_type, index)
            op.parse()
#print(op)

            if op.status.parsed:
                self.operators.append(op)
            else:
                if op.isLegacy:
                    self.legacys.append(op)


        print('Pytorch Model Input size:')
        for i, name in enumerate(self.inputs):
            print(name, self.inputs_shape[i])

        self.setParsed()


    def convert(self):
        logger.debug('Converting the Model...')

        for i, input in enumerate(self.inputs):
            self.layers.append(make_caffe_input_layer(input, self.inputs_shape[i], i, self.param))

        for op in self.operators:
            logger.debug(op)
            layers = op.convert()
#print(layers, op.type)
            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_path):
        save_caffe_model(caffe_path, self.layers)


    def dump(self, model_byte, model_name, input_tensor, dump_level=-1):
        dump = Dump('Pytorch', model_byte, model_name, input_tensor, self.param, dump_level)
        from progress_bar import ProgressBar
        progressBar = ProgressBar(len(self.operators), 0, "Ptyroch dump processing")
        for i, op in enumerate(self.operators):
            dump.operator(op)
            progressBar.setValue(i)
        progressBar.onCancel()