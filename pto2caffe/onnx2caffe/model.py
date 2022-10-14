import sys
import onnx
import logging
import numpy as np
from base import Base


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
from onnx2caffe.op.cast import Cast
from onnx2caffe.op.tanh import TanH
from onnx2caffe.op.tile import Tile
from onnx2caffe.op.power import Pow
from onnx2caffe.op.sqrt import Sqrt
from onnx2caffe.op.relu import ReLU
from onnx2caffe.op.clip import ReLUX
from onnx2caffe.op.floor import Floor
from onnx2caffe.op.prelu import PReLU
from onnx2caffe.op.shape import Shape
from onnx2caffe.op.slice import Slice
from onnx2caffe.op.split import Split
from onnx2caffe.op.where import Where
from onnx2caffe.op.concat import Concat
from onnx2caffe.op.expand import Expand
from onnx2caffe.op.gather import Gather
from onnx2caffe.op.resize import Resize
from onnx2caffe.op.matmul import MatMul
from onnx2caffe.op.compare import Compare
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
from onnx2caffe.op.batchnorm import BatchNorm
from onnx2caffe.op.reducemax import ReduceMax
from onnx2caffe.op.reducemean import ReduceMean
from onnx2caffe.op.hardsigmoid import HardSigmoid
from onnx2caffe.op.convtranspose import Deconvolution
from onnx2caffe.op.globalaveragepool import GlobalAveragePool
from onnx2caffe.op.instancenormalization import InstanceNormalization


from onnx2caffe.op.mish import Mish

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer

numpy_dtype = [None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, 'string', np.bool, np.float16, np.double, np.uint32, np.uint64, np.complex64, np.complex128, 'bfloat16']

logger = logging.getLogger('ONNX2Caffe')

#from onnx2caffe.op.debug import Debug
OpMap = {
    'Add': Add,
    'Div': Div,
    'Elu': Elu,
    'Exp': Exp,
    'Log': Log,
    'LRN': LRN,
    'Mul': Mul,
    'Pad': Pad,
    'Pow': Pow,
    'Sub': Sub,
    'Sum': Sum,
    'Cast': Cast,
    'Relu': ReLU,
    'Sqrt': Sqrt,
    'Tanh': TanH,
    'Tile': Tile,
    'Clip': ReLUX,
    'Floor': Floor,
    'PRelu': PReLU,
    'Shape': Shape,
    'Slice': Slice,
    'Split': Split,
    'Where': Where,
    'Equal': Compare,
    'Concat': Concat,
    'Expand': Expand,
    'Gather': Gather,
    'MatMul': MatMul,
    'Resize': Resize,
    'LeakyRelu': ReLU,
    'Dropout': Dropout,
    'Flatten': Flatten,
    'MaxPool': Pooling,
    'Reshape': Reshape,
    'Sigmoid': Sigmoid,
    'Softmax': Softmax,
    'Squeeze': Reshape,
    'Conv': Convolution,
    'Gemm': InnerProduct,
    'Constant': Constant,
    'Softplus': Softplus,
    'Transpose': Permute,
    'Unsqueeze': Reshape,
    'ReduceMax': ReduceMax,
    'AveragePool': Pooling,
    'ReduceMean': ReduceMean,
    'HardSigmoid': HardSigmoid,
    'ConstantOfShape': Constant,
    'ConvTranspose': Deconvolution,
    'BatchNormalization': BatchNorm,
    'GlobalAveragePool': GlobalAveragePool,
    'InstanceNormalization': InstanceNormalization,
    'Upsample': Upsample, #Deprecated
    'Mish': Mish, # Yolov4
}


class Model(Base):

    def __init__(self, onnx_model, param):
        super().__init__(onnx_model, onnx_model.graph)
        self.param = param
        self.model_version = onnx_model.model_version
        self.producer = onnx_model.producer_name +' '+ onnx_model.producer_version

        self.opset = []
        for i in range(len(onnx_model.opset_import)):
            opset_version = onnx_model.opset_import[i].version
            if opset_version <= 17 and opset_version > 3:
                self.opset.append(opset_version)
            else:
                sys.exit('Error: Model opset > 13 or <= 3, it may cause incompatiblility issue. (opset:{})\n'.format(opset_version))


        self.inputs = list()
        self.inputs_shape = list()
        self.inputs_dtype = list()
        self.inputs_maxval = list()
        self.inputs_minval = list()

        self.outputs = list()
        self.outputs_shape = list()
        self.outputs_dtype = list()
        self.outputs_maxval = list()
        self.outputs_minval = list()

        self.pad = dict()
        self.shape = dict()
        self.layers = list()
        self.constant = dict()
        self.errorMsg = list()
        self.indentity = dict()
        self.operators = list()
        self.unsupport = list()

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
            self.constant[tensor.name] = onnx.numpy_helper.to_array(tensor)
        for tensor in self.model.graph.sparse_initializer:
            self.constant[tensor.name] = onnx.numpy_helper.to_array(tensor)

        if len(self.graph.input) == 0 or self.graph.input is None:
            sys.exit('model input can\'t be None')

        print('ONNX Model Input size: (opset=%d)' %self.opset[0])
        for input in self.graph.input:
            if input.name not in self.constant:
                print(input.name, self.shape[input.name])
                self.inputs.append(input.name)
                self.inputs_shape.append(self.shape[input.name])
                self.inputs_dtype.append(numpy_dtype[input.type.tensor_type.elem_type])
                self.inputs_maxval.append(None)
                self.inputs_minval.append(None)
                if not all(self.shape[input.name]):
                    sys.exit('Error: Dynamic Model input detected, Please Use -input_shape to overwrite input shape.')

        for output in self.graph.output:
            self.outputs.append(output.name)
            self.outputs_shape.append(self.shape[output.name])
            self.outputs_dtype.append(numpy_dtype[output.type.tensor_type.elem_type])
            self.outputs_maxval.append(None)
            self.outputs_minval.append(None)

        for index, node in enumerate(self.graph.node):
            if node.op_type in ['Identity'] and node.name not in self.inputs: #ignore op
                self.indentity[node.output[0]] = node.input[0]
                continue

            if node.op_type not in OpMap: # Unsupport OP
                logger.debug(node.op_type, node.name)
                self.unsupport.append(node.op_type)
                continue

            op = OpMap[node.op_type](self, node, index)
            op.parse()

            if op.status.parsed:
                self.operators.append(op)

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
            self.layers.extend(op.convert())

        self.setConverted()


    def save(self, caffe_model_path):
        return save_caffe_model(caffe_model_path, self.layers)


    def forward(self, output_name, inputs_tensor):
        if output_name[0].find('split') >= 0 or output_name[0].find('useless') >= 0:
            return None

        def onnx_run(model, inputs_tensor):
            import onnxruntime
            onnxruntime.set_default_logger_severity(4)

            if onnxruntime.get_device() == 'GPU':
                onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            if onnxruntime.get_device() == 'CPU':
                onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

            output = list()
            output.extend([node.name for node in onnx_session.get_outputs()])

            input_feed = {}
            for index, name in enumerate(self.inputs):
                input_feed[name] = inputs_tensor[index]

            return onnx_session.run(output, input_feed=input_feed)

        if output_name[0] in self.outputs:
            outputs = onnx_run(self.model, inputs_tensor)
            return outputs[self.outputs.index(output_name[0])]
        elif output_name[0] in self.shape.keys():
            self.model.graph.output.insert(len(self.outputs), onnx.helper.make_tensor_value_info(output_name[0], onnx.TensorProto.FLOAT, self.shape[output_name[0]]))
            return onnx_run(self.model, inputs_tensor)[len(self.outputs)]
        else:
            return None
