import logging
from dump import Dump
from onnx import numpy_helper

from base import Base

from onnx2caffe.op.pad import Pad
from onnx2caffe.op.binary import Binary
from onnx2caffe.op.concat import Concat
from onnx2caffe.op.resize import Resize
from onnx2caffe.op.reshape import Reshape #Not Finish yet
from onnx2caffe.op.pooling import Pooling
from onnx2caffe.op.flatten import Flatten
from onnx2caffe.op.conv import Convolution
from onnx2caffe.op.gemm import InnerProduct
from onnx2caffe.op.constant import Constant
from onnx2caffe.op.batchnorm import BatchNorm
from onnx2caffe.op.activation import Activation
from onnx2caffe.op.upsample import Upsample #Deprecated

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer


logger = logging.getLogger('ONNX2caffe')


OpMap = {
    'Pad': Pad,
    'Mul': Binary,
    'Add': Binary,
    'Concat': Concat,
    'Resize': Resize,
    'Flatten': Flatten,
    'MaxPool': Pooling,
    'Relu': Activation,
    'Conv': Convolution,
    'Gemm': InnerProduct,
    'Constant': Constant,
    'Unsqueeze': Reshape,
    'Sigmoid': Activation,
    'AveragePool': Pooling,
    'LeakyRelu': Activation,
    'GlobalAveragePool': Pooling,
    'BatchNormalization': BatchNorm,
    'Upsample': Upsample, #Deprecated
#    'PAD': Pad,
#    'RESHAPE': Reshape,
#    'SOFTMAX': Softmax,
#    'AVERAGE_POOL_2D': AvgPool2d,
#    'FULLY_CONNECTED': InnerProduct,
#    'DEPTHWISE_CONV_2D': Convolution,
#    'RESIZE_NEAREST_NEIGHBOR': Resize,
}


class Model(Base):
    def __init__(self, onnx_model, param):
        super().__init__(onnx_model, onnx_model.graph)
        self.param = param
        self.operators = []
        self.layers = []
        self.input_tensor = dict()
        self.shape = dict()
        self.legacys = []
        self.setInited()


    def parse(self):
        logger.debug("Parsing the ONNX Model...")

        # Get Shape
        for value_info in self.graph.value_info:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        for value_info in self.graph.input:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        for value_info in self.graph.output:
            self.shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]

        print(self.shape[self.graph.input[0].name], '===========')

        # Get Weight & Bias
        for tensor in self.model.graph.initializer:
            self.input_tensor[tensor.name] =  numpy_helper.to_array(tensor)
        for tensor in self.model.graph.sparse_initializer:
            self.input_tensor[tensor.name] =  numpy_helper.to_array(tensor)

        for index, node in enumerate(self.graph.node):
            op = OpMap[node.op_type](self, node, index)
            op.parse()
            if op.status.parsed:
                self.operators.append(op)
            else:
                self.legacys.append(op)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for input in self.graph.input:
            self.layers.append(make_caffe_input_layer(input.name, self.param))
        for op in self.operators:
            logger.debug(op)
            layers = op.convert()
            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_name, caffe_path):
        save_caffe_model(caffe_name, caffe_path, self.layers)


    def dump(self, onnx_model, model_name, input_tensor, dump_level=-1):
        dump = Dump('onnx', onnx_model, model_name, input_tensor, self.param, dump_level)
        from progress_bar import ProgressBar
        progressBar = ProgressBar(len(self.operators), 0, "TFlite dump processing")
        for i, op in enumerate(self.operators):
            dump.operator(op)
            progressBar.setValue(i)
        progressBar.onCancel()
