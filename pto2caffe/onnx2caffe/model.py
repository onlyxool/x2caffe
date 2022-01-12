import logging
from dump import Dump
from onnx import numpy_helper
from collections import Iterable

from base import Base

from onnx2caffe.op.log import Log
from onnx2caffe.op.exp import Exp
from onnx2caffe.op.pad import Pad
from onnx2caffe.op.lrn import LRN
from onnx2caffe.op.mul import Mul
from onnx2caffe.op.add import Add
from onnx2caffe.op.sub import Sub
from onnx2caffe.op.tanh import TanH
from onnx2caffe.op.power import Pow
from onnx2caffe.op.slice import Slice
from onnx2caffe.op.split import Split
from onnx2caffe.op.reduce import Reduce
from onnx2caffe.op.binary import Binary
from onnx2caffe.op.concat import Concat
from onnx2caffe.op.resize import Resize
from onnx2caffe.op.matmul import MatMul
#from onnx2caffe.op.expand import Expand
from onnx2caffe.op.customop import Mish
from onnx2caffe.op.reshape import Reshape #Not Finish yet
from onnx2caffe.op.pooling import Pooling
from onnx2caffe.op.dropout import Dropout
from onnx2caffe.op.flatten import Flatten
from onnx2caffe.op.conv import Convolution
from onnx2caffe.op.gemm import InnerProduct
from onnx2caffe.op.constant import Constant
from onnx2caffe.op.softplus import Softplus
from onnx2caffe.op.transpose import Permute
from onnx2caffe.op.batchnorm import BatchNorm
from onnx2caffe.op.deconv import Deconvolution
from onnx2caffe.op.activation import Activation
from onnx2caffe.op.upsample import Upsample #Deprecated
from onnx2caffe.op.instancenormalization import InstanceNormalization # Not test yet

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer


logger = logging.getLogger('ONNX2caffe')


OpMap = {
    'Exp': Exp,
    'Log': Log,
    'Pad': Pad,
    'LRN': LRN,
    'Add': Add,
    'Sub': Sub,
    'Mul': Mul,
    'Pow': Pow,
    'Tanh': TanH,
    'Sum': Binary,
    'Div': Binary,
    'Slice': Slice,
    'Split': Split,
    'MatMul': MatMul,
#    'Expand': Expand,
    'Concat': Concat,
    'Resize': Resize,
    'Dropout': Dropout,
    'Reshape': Reshape,
    'Squeeze': Reshape,
    'Flatten': Flatten,
    'MaxPool': Pooling,
    'Relu': Activation,
    'Clip': Activation,
    'Conv': Convolution,
    'PRelu': Activation,
    'Identity': Reshape,
    'Gemm': InnerProduct,
    'Constant': Constant,
    'Softplus': Softplus,
    'Unsqueeze': Reshape,
    'Transpose': Permute,
    'ReduceMean': Reduce,
    'Sigmoid': Activation,
    'Softmax': Activation,
    'AveragePool': Pooling,
    'LeakyRelu': Activation,
    'GlobalAveragePool': Pooling,
    'ConvTranspose': Deconvolution,
    'BatchNormalization': BatchNorm,
    'InstanceNormalization': InstanceNormalization,
    'Upsample': Upsample, #Deprecated
    'Mish': Mish, # Yolov4
#    'PAD': Pad,
#    'RESHAPE': Reshape,
#    'SOFTMAX': Softmax,
#    'ConstantOfShape': Constant,
#    'AVERAGE_POOL_2D': AvgPool2d,
#    'FULLY_CONNECTED': InnerProduct,
#    'DEPTHWISE_CONV_2D': Convolution,
#    'RESIZE_NEAREST_NEIGHBOR': Resize,
}


class Model(Base):

    def __init__(self, onnx_model, param):
        super().__init__(onnx_model, onnx_model.graph)
        self.model_version = onnx_model.model_version
        self.producer = onnx_model.producer_name +' '+ onnx_model.producer_version

        self.opset = []
        for i in range(len(onnx_model.opset_import)):
            opset_version = onnx_model.opset_import[i].version
            if opset_version <= 13:
                self.opset.append(opset_version)
            else:
                raise NotImplementedError('Model opset > 13, it may cause incompatiblility issue.')

        self.param = param
        self.operators = []
        self.layers = []
        self.inputs = []
        self.input_tensor = dict()
        self.shape = dict()
        self.legacys = []
        self.setInited()


    def ReplaceActivation(self, node, op_list, activation):
#        skip_op = ['Constant', 'Reshape']
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
        nodes = self.graph.node

        self.ReplaceActivation(nodes, ['Exp', 'Add' , 'Log', 'Tanh', 'Mul'], 'Mish')
        self.ReplaceActivation(nodes, ['Add', 'Clip' , 'Div', 'Mul'], 'Hardswish')
        self.ReplaceActivation(nodes, ['Sigmoid', 'Mul'], 'Swish')


    def parse(self):
        logger.debug("Parsing the ONNX Model...")

#        self.preprocess()

        # Get Shape
        for value_info in self.graph.value_info:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        for value_info in self.graph.input:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        for value_info in self.graph.output:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]

        # Get Weight & Bias
        for tensor in self.model.graph.initializer:
            self.input_tensor[tensor.name] =  numpy_helper.to_array(tensor)
        for tensor in self.model.graph.sparse_initializer:
            self.input_tensor[tensor.name] =  numpy_helper.to_array(tensor)

        if len(self.graph.input) == 0 or self.graph.input is None:
            import sys
            sys.exit('Model input can\'t be None')

        print('ONNX Model Input size: (opset=%d)' %self.opset[0])
        for input in self.graph.input:
            if input.name not in self.input_tensor:
                self.inputs.append(input.name)
                print(input.name, self.shape[input.name])

        for index, node in enumerate(self.graph.node):
            if node.op_type not in OpMap:
                import sys
                errorMsg = 'Error: Operator [' + node.op_type + '] does not Support.\n'
                sys.exit(errorMsg)

            op = OpMap[node.op_type](self, node, index)
            op.parse()
            if op.status.parsed:
                self.operators.append(op)
            else:
                if op.isLegacy:
                    self.legacys.append(op)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for index, input in enumerate(self.inputs):
            self.layers.append(make_caffe_input_layer(input, self.shape[input], index, self.param))
        for op in self.operators:
            logger.debug(op)
            layers = op.convert()
            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_path):
        save_caffe_model(caffe_path, self.layers)


    def dump(self, onnx_model, model_name, input_tensor, dump_level=-1):
        dump = Dump('onnx', onnx_model, model_name, input_tensor, self.param, dump_level)
        from progress_bar import ProgressBar
        progressBar = ProgressBar(len(self.operators), 0, "ONNX dump processing")
        for i, op in enumerate(self.operators):
            dump.operator(op)
            progressBar.setValue(i)
        progressBar.onCancel()
