import logging
from onnx import numpy_helper
from onnx2caffe.op.constant import Constant
from onnx2caffe.op.binary import Binary
from onnx2caffe.op.concat import Concat
from onnx2caffe.op.resize import Resize
from onnx2caffe.op.pooling import Pooling
from onnx2caffe.op.conv import Convolution
from onnx2caffe.op.batchnorm import BatchNorm
from onnx2caffe.op.activation import Activation 

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer


from base import Base

logger = logging.getLogger('ONNX2caffe')

#def Input(model, node, index, legacy):
#    pass

OpMap = {
    'Constant': Constant,
    'Conv': Convolution,
    'LeakyRelu': Activation,
    'Add': Binary,
    'Concat': Concat,
    'BatchNormalization': BatchNorm,
    'Resize': Resize,
    'Sigmoid': Activation,
    'MaxPool': Pooling,
#    'PAD': Pad,
#    'RESHAPE': Reshape,
#    'SOFTMAX': Softmax,
#    'MUL': Mul,
#    'AVERAGE_POOL_2D': AvgPool2d,
#    'FULLY_CONNECTED': InnerProduct,
#    'DEPTHWISE_CONV_2D': Convolution,
#    'RESIZE_NEAREST_NEIGHBOR': Resize,
}


class Model(Base):
    def __init__(self, onnx_model, param):
        super().__init__(onnx_model, onnx_model.graph)
        self.param = param
        self.graph = onnx_model.graph
        self.operators = []
        self.layers = []
        self.input_tensor = dict()
        self.legacys = []
        self.setInited()
        self.shape = dict()


    def parse(self):
        logger.debug("Parsing the Model...")

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

        for index, node in enumerate(self.graph.node):
            op = OpMap[node.op_type](self, node, index)
            op.parse()
            if op.status.parsed:
                self.operators.append(op)

#            op = opFactory(self.model, self.graph, node, index, legacys)
#            op.parse()
#            if op.status.parsed:
#                self.operators.append(op)
#                if hasattr(op, 'activ_type_code'):
#                    act_op = handleFusedActivation(op)
#                    self.operators.append(act_op)
#            else:
#                legacys.append(op)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for input in self.graph.input:
            self.layers.append(make_caffe_input_layer(input.name, self.param))
        for op in self.operators:
            print(op)
            layers = op.convert()
            for layer in layers:
                self.layers.append(layer)
        self.setConverted()


    def save(self, caffe_name, caffe_path):
        save_caffe_model(caffe_name, caffe_path, self.layers)
