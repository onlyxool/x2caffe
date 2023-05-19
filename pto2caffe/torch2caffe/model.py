import sys
import torch
import logging
import numpy as np

from base_Model import BaseModel
from torch2caffe.torchscript import _expand_and_optimize_ir
from torch2caffe.internal_graph import InternalTorchIRGraph
from torch2caffe.torchir_passes import flatten_graph_input_values, flatten_graph_output_values
from torch2caffe.torchir_passes import remove_getattr_nodes, transform_inplace_ops #generate_tensor_assignment_ops,

from torch2caffe.op.to import To
from torch2caffe.op.add import Add
from torch2caffe.op.exp import Exp
from torch2caffe.op.int import Int
from torch2caffe.op.mul import Mul
from torch2caffe.op.pow import Pow
from torch2caffe.op.sub import Sub
from torch2caffe.op.sum import Sum
from torch2caffe.op.copy import Copy
from torch2caffe.op.full import Full
from torch2caffe.op.mean import Mean
from torch2caffe.op.relu import ReLU
from torch2caffe.op.silu import Silu
from torch2caffe.op.size import Size
from torch2caffe.op.view import View
from torch2caffe.op.tuple import Tuple
from torch2caffe.op.slice import Slice
from torch2caffe.op.stack import Stack
from torch2caffe.op.arange import Arange
from torch2caffe.op.detach import Detach
from torch2caffe.op.concat import Concat
from torch2caffe.op.linear import Linear
from torch2caffe.op.matmul import Matmul
from torch2caffe.op.select import Select
from torch2caffe.op.dropout import Dropout
from torch2caffe.op.flatten import Flatten
from torch2caffe.op.permute import Permute
from torch2caffe.op.reshape import Reshape
from torch2caffe.op.pooling import Pooling
from torch2caffe.op.sigmoid import Sigmoid
from torch2caffe.op.softmax import Softmax
from torch2caffe.op.constant import Constant
from torch2caffe.op.meshgrid import Meshgrid
from torch2caffe.op.transpose import Transpose
from torch2caffe.op.unsqueeze import Unsqueeze
from torch2caffe.op.batchnorm import BatchNorm
from torch2caffe.op.contiguous import Contiguous
from torch2caffe.op.listunpack import Listunpack
from torch2caffe.op.convolution import Convolution
from torch2caffe.op.numtotensor import Numtotensor
from torch2caffe.op.floor_divide import Floor_divide
from torch2caffe.op.constantchunk import Constantchunk
from torch2caffe.op.listconstruct import Listconstruct
from torch2caffe.op.constant_pad_nd import ConstantPad
from torch2caffe.op.upsamplenearest import UpsampleNearest
from torch2caffe.op.upsamplebilinear import UpsampleBilinear
from torch2caffe.op.adaptive_avg_pool2d import AdaptiveAvgPooling

logger = logging.getLogger('Torch2Caffe')

passes = [transform_inplace_ops, flatten_graph_input_values, flatten_graph_output_values, remove_getattr_nodes]#, generate_tensor_assignment_ops]


OpMap = {
    'to': To,
    'add': Add,
    'exp': Exp,
    'int': Int,
    'mul': Mul,
    'pow': Pow,
    'sub': Sub,
    'sum': Sum,
    'copy': Copy,
    'full': Full,
    'mean': Mean,
    'relu': ReLU,
    'silu': Silu,
    'size': Size,
    'view': View,
    'cat': Concat,
    'slice': Slice,
    'stack': Stack,
    'arange': Arange,
    'detach': Detach,
    'linear': Linear,
    'matmul': Matmul,
    'select': Select,
    'dropout': Dropout,
    'flatten': Flatten,
    'permute': Permute,
    'reshape': Reshape,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'constant': Constant,
    'meshgrid': Meshgrid,
    'tupleunpack': Tuple,
    'avg_pool2d': Pooling,
    'max_pool2d': Pooling,
    'transpose': Transpose,
    'unsqueeze': Unsqueeze,
    'batch_norm': BatchNorm,
    'tupleconstruct': Tuple,
    'contiguous': Contiguous,
    'listunpack': Listunpack,
    'convolution': Convolution,
    'numtotensor': Numtotensor,
    'floor_divide': Floor_divide,
    'constantchunk': Constantchunk,
    'listconstruct': Listconstruct,
    'constant_pad_nd': ConstantPad,
    'upsample_nearest2d': UpsampleNearest,
    'upsample_bilinear2d': UpsampleBilinear,
    'adaptive_avg_pool2d': AdaptiveAvgPooling,
}


class Model(BaseModel):
    def __init__(self, model, param):
        raw_graph, params_dict = _expand_and_optimize_ir(model)
        graph = InternalTorchIRGraph(raw_graph, params_dict, torch.rand(param['input_shape']), None)
        [p(graph) for p in passes]
        super().__init__(model, graph, param)

        if param['log'] <= 1:
            print(graph)

        # Inputs
        self.constant = self.graph.params
        self.inputs = list(self.graph.inputs.keys())
        self.inputs_shape = [list(param['input_shape'])]
        self.inputs_dtype = [np.float32 for i in range(len(self.inputs_shape))]
        for index, input_name in enumerate(self.inputs):
            self.inputs_maxval.append(None)
            self.inputs_minval.append(None)

        # Outputs
        for output in self.graph.outputs:
            self.outputs.append(output)
            self.outputs_maxval.append(None)
            self.outputs_minval.append(None)

        # Shape
        for index, input_name in enumerate(self.inputs):
            self.tensor_shape[input_name] = self.inputs_shape[index]

        # Constants
        for key, value in self.constant.items():
            self.tensor_shape[key] = list(value.shape)
            self.tensor_dtype[key] = value.dtype

        self.setInited()


    def parse(self):
        logger.debug('Parsing the Pytorch Model...')
        print('Pytorch Model Input Size:')
        for index, input_name in enumerate(self.graph.inputs):
            print(self.inputs[index], self.inputs_shape[index])

        for index, node in enumerate(self.graph.nodes):
            node.kind = node.kind.strip('_')
            if node.kind not in OpMap: # Unsupport OP
                from torch2caffe.op.operator import Operator
                op = Operator(self, self.graph, node, index)
                op.__parse__()
                self.unsupport.append(node.kind)

                if self.param['log'] <= 1:
                    print(op)
                    print(node)

                continue

            op = OpMap[node.kind](self, node, index)
            op.parse()

            if op.status.parsed:
                output = op.forward()
                op.post_forward(output if isinstance(output, tuple) else [output])
                self.operators.append(op)

        for errorMsg in list(set(self.errorMsg)):
            print(errorMsg)

        if len(self.unsupport) > 0:
            errorMsg = 'Error: Operator ' + str(list(set(self.unsupport))) + ' does not Support.\n'
            sys.exit(errorMsg)

        self.setParsed()


    def forward(self, output_name, inputs_tensor):
        if output_name.find('split') >= 0 or output_name.find('intermediate') >= 0:
            return None

        return self.variable[output_name].cpu().detach().numpy() if self.variable[output_name].is_cuda else self.variable[output_name].detach().numpy()
