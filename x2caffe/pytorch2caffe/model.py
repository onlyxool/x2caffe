import sys
import torch
import logging
import numpy as np
from base_Model import BaseModel

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

from pytorch2caffe.pnnx import Pnnx

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

#'aten::stack': Debug,
#'pnnx.Attribute': Debug,
}

ignore_op = ['prim::TupleConstruct']

class Model(BaseModel):

    def __init__(self, pytorch_file, param):#, inputs_tensor):
        super().__init__(None, None, param)
        self.pytorch_file = pytorch_file
        self.device = torch.device('cpu')#torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inputs_shape = [list(param['input_shape'])]
        self.inputs_dtype = [np.float32 for i in range(len(self.inputs_shape))]

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


    def forward(self, output_name, inputs_tensor):
        if not self.status.forwarded:
            self.pnnx.model_forward(inputs_tensor[0])
            self.setForwarded()

        for op in self.operators:
            if op.name == output_name:
                return self.pnnx.get_ops_output(output_name, np.prod(op.outputs_shape[0]))
