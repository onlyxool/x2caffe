import sys
import logging
import numpy as np
from base import Base
from onnx import numpy_helper


from onnx2caffe.op.elu import Elu
from onnx2caffe.op.exp import Exp
from onnx2caffe.op.log import Log
from onnx2caffe.op.pad import Pad
from onnx2caffe.op.lrn import LRN
from onnx2caffe.op.mul import Mul
from onnx2caffe.op.add import Add
from onnx2caffe.op.sum import Sum
from onnx2caffe.op.sub import Sub
from onnx2caffe.op.div import Div
from onnx2caffe.op.tanh import TanH
from onnx2caffe.op.power import Pow
from onnx2caffe.op.sqrt import Sqrt
from onnx2caffe.op.relu import ReLU
from onnx2caffe.op.clip import ReLUX
from onnx2caffe.op.prelu import PReLU
from onnx2caffe.op.slice import Slice
from onnx2caffe.op.split import Split
from onnx2caffe.op.concat import Concat
from onnx2caffe.op.resize import Resize
from onnx2caffe.op.matmul import MatMul
from onnx2caffe.op.sigmoid import Sigmoid
from onnx2caffe.op.softmax import Softmax
from onnx2caffe.op.reshape import Reshape
from onnx2caffe.op.pooling import Pooling
from onnx2caffe.op.dropout import Dropout
from onnx2caffe.op.flatten import Flatten
from onnx2caffe.op.conv import Convolution
from onnx2caffe.op.gemm import InnerProduct
from onnx2caffe.op.upsample import Upsample
from onnx2caffe.op.constant import Constant
from onnx2caffe.op.softplus import Softplus
from onnx2caffe.op.transpose import Permute
from onnx2caffe.op.reducemean import Reduce
from onnx2caffe.op.batchnorm import BatchNorm
from onnx2caffe.op.convtranspose import Deconvolution
from onnx2caffe.op.instancenormalization import InstanceNormalization

from onnx2caffe.op.debug import Debug

from onnx2caffe.op.mish import Mish

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer

numpy_dtype = [None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, 'string', np.bool, np.float16, np.double, np.uint32, np.uint64, np.complex64, np.complex128, 'bfloat16']

logger = logging.getLogger('ONNX2Caffe')

OpMap = {
    'Elu': Elu,
    'Exp': Exp,
    'Log': Log,
    'Pad': Pad,
    'LRN': LRN,
    'Add': Add,
    'Sum': Sum,
    'Sub': Sub,
    'Mul': Mul,
    'Div': Div,
    'Pow': Pow,
    'Tanh': TanH,
    'Sqrt': Sqrt,
    'Relu': ReLU,
    'Clip': ReLUX,
    'PRelu': PReLU,
    'Slice': Slice,
    'Split': Split,
    'MatMul': MatMul,
    'Concat': Concat,
    'Resize': Resize,
    'LeakyRelu': ReLU,
    'Sigmoid': Sigmoid,
    'Softmax': Softmax,
    'Dropout': Dropout,
    'Reshape': Reshape,
    'Squeeze': Reshape,
    'Flatten': Flatten,
    'MaxPool': Pooling,
    'Conv': Convolution,
    'Identity': Reshape,
    'Gemm': InnerProduct,
    'Constant': Constant,
    'Softplus': Softplus,
    'Unsqueeze': Reshape,
    'Transpose': Permute,
    'ReduceMean': Reduce,
    'AveragePool': Pooling,
    'GlobalAveragePool': Pooling,
    'ConvTranspose': Deconvolution,
    'BatchNormalization': BatchNorm,
    'InstanceNormalization': InstanceNormalization,
    'Upsample': Upsample, #Deprecated
    'Mish': Mish, # Yolov4

#    'Multinomial': Debug,
#    'Expand': Debug,
#    'ConstantOfShape': Debug,
}


class Model(Base):

    def __init__(self, onnx_model, param):
        super().__init__(onnx_model, onnx_model.graph)
        self.model_version = onnx_model.model_version
        self.producer = onnx_model.producer_name +' '+ onnx_model.producer_version

        self.opset = []
        for i in range(len(onnx_model.opset_import)):
            opset_version = onnx_model.opset_import[i].version
            if opset_version <= 13 and opset_version > 3:
                self.opset.append(opset_version)
            else:
                sys.exit('Error: Model opset > 13 or <= 3, it may cause incompatiblility issue. (opset:{})\n'.format(opset_version))

        self.param = param
        self.inputs = list()
        self.inputs_shape = list()
        self.inputs_dtype = list()
        self.inputs_maxval = list()
        self.inputs_minval = list()
        self.constant = dict()
        self.constant = dict()
        self.operators = list()
        self.unsupport = list()
        self.errorMsg = list()
        self.layers = list()
        self.shape = dict()
        self.legacys = list()
        self.setInited()


    def ReplaceActivation(self, node, op_list, activation):
        skip_op = []
        for i in range(len(node)):
            if i >= len(node):
                break
            isflag = True
            cnt = 0
            for j in range(len(op_list)):
                if (i+j+cnt>=len(node)) or (node[i+j+cnt].op_type != op_list[j]):
                    isflag = False
                    break

                while (i+j+cnt+1 < len(node)) and  node[i+j+cnt+1].op_type in skip_op:
                    cnt+=1

            if(isflag):
                node[i].output[0] = node[i+len(op_list)-1+cnt].output[0]
                node[i].op_type = activation
                for j in range(len(op_list) - 1 + cnt):
                    node.remove(node[i+1])


    def preprocess(self):
#        nodes = self.graph.node
#        self.ReplaceActivation(nodes, ['Exp', 'Add' , 'Log', 'Tanh', 'Mul'], 'Mish')
#        self.ReplaceActivation(nodes, ['Add', 'Clip' , 'Div', 'Mul'], 'Hardswish')
#        self.ReplaceActivation(nodes, ['Sigmoid', 'Mul'], 'Swish')

        if self.graph.node[0].op_type == 'Transpose' and self.model.producer_name == 'tf2onnx':
            print(self.model.producer_name)
            for attr in self.graph.node[0].attribute:
                if attr.name == 'perm' and list(attr.ints) == [0, 3, 1, 2]:
                    if self.graph.node[1].input[0] == self.graph.node[0].output[0]:
                        self.graph.node[1].input[0] = self.graph.node[0].input[0]

                    self.graph.node.remove(self.graph.node[0])
                    self.layout = 'NHWC'


    def parse(self):
        logger.debug("Parsing the ONNX Model...")

        # Get Shape
        for value_info in self.graph.value_info:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        for value_info in self.graph.input:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        for value_info in self.graph.output:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]

        # Get Weight & Bias
        for tensor in self.model.graph.initializer:
            self.constant[tensor.name] =  numpy_helper.to_array(tensor)
        for tensor in self.model.graph.sparse_initializer:
            self.constant[tensor.name] =  numpy_helper.to_array(tensor)

        if len(self.graph.input) == 0 or self.graph.input is None:
            sys.exit('model input can\'t be none')

        print('ONNX Model Input size: (opset=%d)' %self.opset[0])
        for input in self.graph.input:
            if input.name not in self.constant:
                print(input.name, self.shape[input.name])

                self.inputs.append(input.name)
                self.inputs_shape.append(self.shape[input.name])
                self.inputs_dtype.append(numpy_dtype[input.type.tensor_type.elem_type])
                self.inputs_maxval.append(None)
                self.inputs_minval.append(None)

        self.param['inputs_shape'] = self.inputs_shape

        for index, node in enumerate(self.graph.node):
            if node.op_type not in OpMap:
                self.unsupport.append(node.op_type)
                continue

            op = OpMap[node.op_type](self, node, index)
            op.parse()

            if op.status.parsed:
                self.operators.append(op)
            elif op.isLegacy:
                self.legacys.append(op)

        for errorMsg in list(set(self.errorMsg)):
            print(errorMsg)

        if len(self.unsupport) > 0:
            errorMsg = 'Error: Operator ' + str(list(set(self.unsupport))) + ' does not Support.\n'
            sys.exit(errorMsg)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for index, input in enumerate(self.inputs):
            self.layers.append(make_caffe_input_layer(input, self.shape[input], index, self.param))

        for op in self.operators:
            layers = op.convert()
            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_model_path):
        save_caffe_model(caffe_model_path, self.layers)
