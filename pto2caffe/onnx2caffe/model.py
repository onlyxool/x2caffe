import sys
import onnx
import logging
import numpy as np
from base_Model import BaseModel
from util import isShapeFullyDefined

from onnx2caffe.op.abs import Abs
from onnx2caffe.op.add import Add
from onnx2caffe.op.div import Div
from onnx2caffe.op.elu import Elu
from onnx2caffe.op.exp import Exp
from onnx2caffe.op.log import Log
from onnx2caffe.op.lrn import LRN
from onnx2caffe.op.mul import Mul
from onnx2caffe.op.pad import Pad
from onnx2caffe.op.sub import Sub
from onnx2caffe.op.sum import Sum
from onnx2caffe.op.cast import Cast
from onnx2caffe.op.less import Less
from onnx2caffe.op.relu import ReLU
from onnx2caffe.op.sqrt import Sqrt
from onnx2caffe.op.tanh import TanH
from onnx2caffe.op.tile import Tile
from onnx2caffe.op.power import Pow
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
from onnx2caffe.op.matmul import MatMul
from onnx2caffe.op.resize import Resize
from onnx2caffe.op.compare import Compare
from onnx2caffe.op.dropout import Dropout
from onnx2caffe.op.flatten import Flatten
from onnx2caffe.op.pooling import Pooling
from onnx2caffe.op.nonzero import NonZero
from onnx2caffe.op.reshape import Reshape
from onnx2caffe.op.sigmoid import Sigmoid
from onnx2caffe.op.softmax import Softmax
from onnx2caffe.op.conv import Convolution
from onnx2caffe.op.gemm import InnerProduct
from onnx2caffe.op.constant import Constant
from onnx2caffe.op.softplus import Softplus
from onnx2caffe.op.upsample import Upsample
from onnx2caffe.op.transpose import Permute
from onnx2caffe.op.batchnorm import BatchNorm
from onnx2caffe.op.reducemax import ReduceMax
from onnx2caffe.op.reducesum import ReduceSum
from onnx2caffe.op.unsqueeze import Unsqueeze
from onnx2caffe.op.reducemean import ReduceMean
from onnx2caffe.op.hardsigmoid import HardSigmoid
from onnx2caffe.op.convtranspose import Deconvolution
from onnx2caffe.op.globalaveragepool import GlobalAveragePool
from onnx2caffe.op.instancenormalization import InstanceNormalization

from onnx2caffe.op.mish import Mish


numpy_dtype = [None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, 'string', np.bool_, np.float16, np.double, np.uint32, np.uint64, np.complex64, np.complex128, 'bfloat16']

logger = logging.getLogger('ONNX2Caffe')

OpMap = {
    'Abs': Abs,
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
    'Less': Less,
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
    'NonZero': NonZero,
    'Reshape': Reshape,
    'Sigmoid': Sigmoid,
    'Softmax': Softmax,
    'Squeeze': Reshape,
    'Conv': Convolution,
    'Gemm': InnerProduct,
    'Constant': Constant,
    'Softplus': Softplus,
    'Upsample': Upsample,
    'Transpose': Permute,
    'ReduceMax': ReduceMax,
    'ReduceSum': ReduceSum,
    'Unsqueeze': Unsqueeze,
    'AveragePool': Pooling,
    'ReduceMean': ReduceMean,
    'HardSigmoid': HardSigmoid,
    'ConstantOfShape': Constant,
    'ConvTranspose': Deconvolution,
    'BatchNormalization': BatchNorm,
    'GlobalAveragePool': GlobalAveragePool,
    'InstanceNormalization': InstanceNormalization,
    'Mish': Mish, # Yolov4
}


class Model(BaseModel):

    def __init__(self, onnx_model, param):
        super().__init__(onnx_model, onnx_model.graph, param)

        self.model_version = onnx_model.model_version
        self.producer = onnx_model.producer_name +' '+ onnx_model.producer_version

        if onnx_model.producer_name == 'onnx.quantize':
            sys.exit('Error: Quantize Model dose not support.\n')

        self.opset = []
        for i in range(len(onnx_model.opset_import)):
            if onnx_model.opset_import[i].domain == '':
                opset_version = onnx_model.opset_import[i].version
                if opset_version <= 17:
                    self.opset.append(opset_version)
                else:
                    sys.exit('Error: Model opset > 17, it may cause incompatiblility issue. (opset:{})\n'.format(opset_version))

        self.setInited()


    def parse(self):
        logger.debug("Parsing the ONNX Model...")

        # Get Shape
        for value_info in self.graph.value_info:
            self.tensor_dtype[value_info.name] = value_info.type.tensor_type.elem_type
            self.tensor_shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        for value_info in self.graph.input:
            self.tensor_dtype[value_info.name] = value_info.type.tensor_type.elem_type
            self.tensor_shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        for value_info in self.graph.output:
            self.tensor_dtype[value_info.name] = value_info.type.tensor_type.elem_type
            self.tensor_shape[value_info.name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]

        for key, value in self.tensor_shape.items():
            self.tensor_shape[key] = self.tensor_shape[key] if isShapeFullyDefined(value) else []

        # Get Weight & Bias
        for tensor in self.model.graph.initializer:
            self.constant[tensor.name] = onnx.numpy_helper.to_array(tensor)
            self.tensor_dtype[tensor.name] = numpy_dtype.index(onnx.numpy_helper.to_array(tensor).dtype)
        for tensor in self.model.graph.sparse_initializer:
            self.constant[tensor.name] = onnx.numpy_helper.to_array(tensor)
            self.tensor_dtype[tensor.name] = numpy_dtype.index(onnx.numpy_helper.to_array(tensor).dtype)

        if len(self.graph.input) == 0 or self.graph.input is None:
            sys.exit('model input can\'t be None')

        print('ONNX Model Input size: (opset=%d)' %self.opset[0])
        for input in self.graph.input:
            if input.name not in self.constant:
                print(input.name, self.tensor_shape[input.name])
                self.inputs.append(input.name)
                self.inputs_shape.append(self.tensor_shape[input.name])
                self.inputs_dtype.append(numpy_dtype[input.type.tensor_type.elem_type])
                self.inputs_maxval.append(None)
                self.inputs_minval.append(None)
                if not all(self.tensor_shape[input.name]):
                    sys.exit('Error: Dynamic Model input detected, Please Use -input_shape to overwrite input shape.')

        for output in self.graph.output:
            self.outputs.append(output.name)
            self.outputs_shape.append(self.tensor_shape[output.name])
            self.outputs_dtype.append(numpy_dtype[output.type.tensor_type.elem_type])
            self.outputs_maxval.append(None)
            self.outputs_minval.append(None)

        for index, node in enumerate(self.graph.node):
            if node.op_type in ['Identity'] and node.name not in self.inputs: #ignore op
                self.indentity[node.output[0]] = node.input[0]
                continue

            if node.op_type not in OpMap: # Unsupport OP
                if self.param['log'] == 1:
                    from onnx2caffe.op.operator import Operator
                    op = Operator(self, node, index)
                    op.__parse__()
                    print(op)
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


    def forward(self, output_name, inputs_tensor):
        if output_name.find('intermediate') >= 0:
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

        if output_name in self.outputs:
            outputs = onnx_run(self.model, inputs_tensor)
            return outputs[self.outputs.index(output_name)]
        elif output_name in self.tensor_shape.keys():
            self.model.graph.output.insert(len(self.outputs), onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, self.tensor_shape[output_name]))
            return onnx_run(self.model, inputs_tensor)[len(self.outputs)]
        else:
            return None
