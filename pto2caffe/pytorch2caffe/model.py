import sys
import torch
import logging
from dump import Dump
from base import Base

from pytorch2caffe.pnnx import Pnnx
from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer

from pytorch2caffe.op.pad import Pad
from pytorch2caffe.op.silu import Silu
from pytorch2caffe.op.relu import ReLU 
from pytorch2caffe.op.mean import Mean
from pytorch2caffe.op.input import Input 
from pytorch2caffe.op.slice import Slice
from pytorch2caffe.op.swish import Swish
from pytorch2caffe.op.concat import Concat
from pytorch2caffe.op.linear import Linear
from pytorch2caffe.op.select import Select
from pytorch2caffe.op.matmul import MatMul
from pytorch2caffe.op.output import Output
from pytorch2caffe.op.pooling import Pooling
from pytorch2caffe.op.flatten import Flatten
from pytorch2caffe.op.permute import Permute
from pytorch2caffe.op.reshape import Reshape
from pytorch2caffe.op.softmax import Softmax
from pytorch2caffe.op.dropout import Dropout
from pytorch2caffe.op.sigmoid import Sigmoid
from pytorch2caffe.op.conv import Convolution
from pytorch2caffe.op.upsample import Upsample
from pytorch2caffe.op.batchnorm import BatchNorm
from pytorch2caffe.op.transpose import Transpose
from pytorch2caffe.op.unsqueeze import Unsqueeze
from pytorch2caffe.op.expression import Expression
from pytorch2caffe.op.convtranspose import Deconvolution
from pytorch2caffe.op.adaptavgpooling import AdaptiveAvgPooling

from pytorch2caffe.op.debug import Debug


logger = logging.getLogger('Pytorch2Caffe')


OpMap = {
    'F.pad': Pad,
    'F.silu': Silu,
    'F.relu': ReLU,
    'nn.ReLU': ReLU,
    'torch.mean': Mean,
    'torch.cat': Concat,
    'nn.Linear': Linear,
    'pnnx.Input': Input,
    'aten::relu_': ReLU,
    'F.softmax': Softmax,
    'F.sigmoid': Sigmoid,
    'nn.Dropout': Dropout,
    'nn.Softmax': Softmax,
    'nn.Sigmoid': Sigmoid,
    'pnnx.Output': Output,
    'nn.Hardswish': Swish,
    'Tensor.slice': Slice,
    'F.upsample': Upsample,
    'Tensor.view': Reshape,
    'aten::matmul': MatMul,
    'nn.Upsample': Upsample,
    'nn.AvgPool2d': Pooling,
    'nn.MaxPool2d': Pooling,
    'Tensor.select': Select,
    'nn.Conv2d': Convolution,
    'F.hardsigmoid': Sigmoid,
    'torch.flatten': Flatten,
    'torch.permute': Permute,
    'Tensor.reshape': Reshape,
    'nn.BatchNorm2d': BatchNorm,
    'torch.transpose': Transpose,
    'torch.unsqueeze': Unsqueeze,
    'pnnx.Expression': Expression,
    'nn.ConvTranspose2d': Deconvolution,
    'nn.AdaptiveAvgPool2d': AdaptiveAvgPooling,
    'F.adaptive_avg_pool2d': AdaptiveAvgPooling,

    'pnnx.Attribute': Debug,
}

ignore_op = ['prim::TupleConstruct']

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
            if op_type in ignore_op:
                continue

            if op_type not in OpMap:
                errorMsg = 'Error: Operator [' + op_type + '] does not Support.\n'
                sys.exit(errorMsg)

            op = OpMap[op_type](self, self.pnnx, op_type, index)
            op.parse()

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

            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_model_path):
        save_caffe_model(caffe_model_path, self.layers)


    def dump(self, model_byte, model_name, input_tensor, dump_level=-1):
        dump = Dump('Pytorch', model_byte, model_name, input_tensor, self.param, dump_level)
        from progress_bar import ProgressBar
        progressBar = ProgressBar(len(self.operators), 0, "Ptyroch dump processing")
        for i, op in enumerate(self.operators):
            dump.operator(op)
            progressBar.setValue(i)
        progressBar.onCancel()
