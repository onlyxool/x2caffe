import sys
import logging
import numpy as np
from base import Base
import tensorflow as tf

from tensorflow2caffe.op.abs import Abs
from tensorflow2caffe.op.add import Add
from tensorflow2caffe.op.log import Log
from tensorflow2caffe.op.max import Max
from tensorflow2caffe.op.mul import Mul
from tensorflow2caffe.op.neg import Neg
from tensorflow2caffe.op.pad import Pad
from tensorflow2caffe.op.pow import Pow
from tensorflow2caffe.op.sub import Sub
from tensorflow2caffe.op.sum import Sum
from tensorflow2caffe.op.exp import Exp
from tensorflow2caffe.op.cast import Cast
from tensorflow2caffe.op.fill import Fill
from tensorflow2caffe.op.loop import Loop
from tensorflow2caffe.op.mean import Mean
from tensorflow2caffe.op.pack import Pack
from tensorflow2caffe.op.prod import Prod
from tensorflow2caffe.op.pool import Pool
from tensorflow2caffe.op.rank import Rank
from tensorflow2caffe.op.relu import ReLU
from tensorflow2caffe.op.size import Size
from tensorflow2caffe.op.sqrt import Sqrt
from tensorflow2caffe.op.tanh import Tanh
from tensorflow2caffe.op.tile import Tile
from tensorflow2caffe.op.enter import Enter
from tensorflow2caffe.op.floor import Floor
from tensorflow2caffe.op.merge import Merge
from tensorflow2caffe.op.range import Range
from tensorflow2caffe.op.relux import ReLUX
from tensorflow2caffe.op.rsqrt import Rsqrt
from tensorflow2caffe.op.shape import Shape
from tensorflow2caffe.op.slice import Slice
from tensorflow2caffe.op.split import Split
from tensorflow2caffe.op.concat import Concat
from tensorflow2caffe.op.select import Select
from tensorflow2caffe.op.splitv import SplitV
from tensorflow2caffe.op.square import Square
from tensorflow2caffe.op.switch import Switch
from tensorflow2caffe.op.matmul import MatMul
from tensorflow2caffe.op.unpack import Unpack
from tensorflow2caffe.op.random import Random
from tensorflow2caffe.op.gather import GatherV2
from tensorflow2caffe.op.biasadd import BiasAdd
from tensorflow2caffe.op.compare import Compare
from tensorflow2caffe.op.logical import Logical
from tensorflow2caffe.op.maximum import Maximum
from tensorflow2caffe.op.minimum import Minimum
from tensorflow2caffe.op.realdiv import RealDiv
from tensorflow2caffe.op.reshape import Reshape
from tensorflow2caffe.op.sigmoid import Sigmoid
from tensorflow2caffe.op.softmax import Softmax
from tensorflow2caffe.op.squeeze import Squeeze
from tensorflow2caffe.op.softplus import Softplus
from tensorflow2caffe.op.conv2d import Convolution
from tensorflow2caffe.op.batchnorm import BatchNorm
from tensorflow2caffe.op.leakyrelu import LeakyRelu
from tensorflow2caffe.op.transpose import Transpose
from tensorflow2caffe.op.expanddims import ExpandDims
from tensorflow2caffe.op.tensorlist import TensorList
from tensorflow2caffe.op.tensorarray import TensorArray
from tensorflow2caffe.op.spacetodepth import SpaceToDepth
from tensorflow2caffe.op.stridedslice import StridedSlice
from tensorflow2caffe.op.resizebilinear import ResizeBilinear
from tensorflow2caffe.op.depthwise import ConvolutionDepthwise
from tensorflow2caffe.op.maxpoolwithargmax import MaxPoolWithArgmax
from tensorflow2caffe.op.conv2dbackpropinput import Conv2DBackpropInput
from tensorflow2caffe.op.resizenearestneighbor import ResizeNearestNeighbor
from tensorflow2caffe.op.placeholderwithdefault import PlaceholderWithDefault
from tensorflow2caffe.op.sparsetensordensematmul import SparseTensorDenseMatMul

from tensorflow2caffe.op.debug import Debug

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer
from util import shape_map_nhwc2nchw, shape_map_nchw2nhwc





logger = logging.getLogger('TensorFlow2Caffe')

OpMap = {
#    'ExtractImagePatches': Debug,
#    'Size': Debug,
#    'SpaceToBatchND': Debug,
#    'Snapshot': Debug,
#    'MatrixBandPart': Debug,
#    'Pow': Debug,
#'All': Debug,

    'Abs': Abs,
    'Add': Add,
    'Exp': Exp,
    'Log': Log,
    'Max': Max,
    'Mul': Mul,
    'Neg': Neg,
    'Pad': Pad,
    'Pow': Pow,
    'Sub': Sub,
    'Sum': Sum,
    'AddN': Add,
    'Cast': Cast,
    'Fill': Fill,
    'Mean': Mean,
    'Pack': Pack,
    'Prod': Prod,
    'Rank': Rank,
    'Relu': ReLU,
    'Size': Size,
    'Sqrt': Sqrt,
    'Tanh': Tanh,
    'Tile': Tile,
    'AddV2': Add,
    'Enter': Enter,
    'Floor': Floor,
    'Merge': Merge,
    'Range': Range,
    'Relu6': ReLUX,
    'Rsqrt': Rsqrt,
    'Shape': Shape,
    'Slice': Slice,
    'Split': Split,
    'Less': Compare,
    'AvgPool': Pool,
    'MaxPool': Pool,
    'Equal': Compare,
    'SplitV': SplitV,
    'Square': Square,
    'Switch': Switch,
    'MatMul': MatMul,
    'Unpack': Unpack,
    'LoopCond': Loop,
    'BiasAdd': BiasAdd,
    'Greater': Compare,
    'Maximum': Maximum,
    'Minimum': Minimum,
    'RealDiv': RealDiv,
    'Reshape': Reshape,
    'Sigmoid': Sigmoid,
    'Softmax': Softmax,
    'Squeeze': Squeeze,
    'ConcatV2': Concat,
    'SelectV2': Select,
    'GatherV2': GatherV2,
    'LogicalOr': Logical,
    'Softplus': Softplus,
    'LogicalAnd': Logical,
    'NextIteration': Loop,
    'Conv2D': Convolution,
    'LeakyRelu': LeakyRelu,
    'Transpose': Transpose,
    'GreaterEqual': Compare,
    'RandomUniform': Random,
    'ExpandDims': ExpandDims,
    'FusedBatchNorm': BatchNorm,
    'SpaceToDepth': SpaceToDepth,
    'StridedSlice': StridedSlice,
    'TensorArrayV3': TensorArray,
    'FusedBatchNormV3': BatchNorm,
    'RandomStandardNormal': Random,
    'TensorListReserve': TensorList,
    'ResizeBilinear': ResizeBilinear,
    'TensorArraySizeV3': TensorArray,
    'TensorArrayReadV3': TensorArray,
    'TensorArrayWriteV3': TensorArray,
    'TensorArrayScatterV3': TensorArray,
    'MaxPoolWithArgmax': MaxPoolWithArgmax,
    'Conv2DBackpropInput': Conv2DBackpropInput,
    'DepthwiseConv2dNative': ConvolutionDepthwise,
    'ResizeNearestNeighbor': ResizeNearestNeighbor,
    'PlaceholderWithDefault': PlaceholderWithDefault,
    'SparseTensorDenseMatMul': SparseTensorDenseMatMul,
}


class Model(Base):

    def __init__(self, model, graph, param):
        super().__init__(model, graph)
        self.param = param
        self.layout = param['layout']
        self.inputs = list()
        self.inputs_shape = list()
        self.inputs_dtype = list()
        self.inputs_maxval = list()
        self.inputs_minval = list()
        self.constant = dict()
        self.indentity = dict()
        self.pad = dict()
        self.operations = list()
        self.operators = list()
        self.unsupport = list()
        self.layers = list()
        self.setInited()


    def preprocess(self, operations):
        # Constant folding
        tf.print('Model Preprocessing', end=' ')
        constant_fold_op_level = list()
        if self.param['optimizify'] >= 1:
            constant_fold_op_level.extend(['Shape', 'Size', 'Reshape', 'StridedSlice', 'Unpack'])
        if self.param['optimizify'] >= 2:
            constant_fold_op_level.extend(['RandomStandardNormal'])
        if self.param['optimizify'] >= 3:
            constant_fold_op_level.extend(['ExpandDims', 'Tile', 'Transpose', 'Pack'])
        if self.param['optimizify'] >= 4:
            constant_fold_op_level.extend(['Add', 'Sub', 'Mul', 'Pow', 'Sqrt', 'FloorDiv', 'FloorMod'])
        if self.param['optimizify'] >= 5:
            constant_fold_op_level.extend(['ArgMax', 'GatherNd', 'Maximum', 'ConcatV2'])
        if self.param['optimizify'] >= 6:
            constant_fold_op_level.extend(['TensorArrayV3', 'TensorArrayReadV3', 'LoopCond', 'NextIteration', 'TensorArrayWriteV3'])

        for op in operations[:]:
            if op.type in constant_fold_op_level:
                inputs_tensor1 = list()
                inputs_tensor2 = list()

                for input_shape in self.inputs_shape:
                    inputs_tensor1.append(np.random.random(shape_map_nchw2nhwc(input_shape)).astype(self.param['dtype']))
                    inputs_tensor2.append(np.random.random(shape_map_nchw2nhwc(input_shape)).astype(self.param['dtype']))

                output1 = self.forward(inputs_tensor1, self.inputs, [op.outputs[0].name])
                output2 = self.forward(inputs_tensor2, self.inputs, [op.outputs[0].name])

                if output1 is not None and output2 is not None and np.allclose(output1, output2):
                    self.constant[op.outputs[0].name] = output1
                    operations.remove(op)
                    tf.print('.', end='')

        print('\n')
        return operations


    def parse_input(self, operations):
        for op in operations:
            if op.type == 'Placeholder' and 'unused_control_flow_input' not in op.outputs[0].name and op.outputs[0].shape.is_fully_defined():
                dtype = type(op.get_attr('dtype').as_numpy_dtype()) if op.get_attr('dtype').is_numpy_compatible else None
                input_shape = shape_map_nhwc2nchw(op.outputs[0].shape.as_list()) if self.layout == 'NHWC' else op.outputs[0].shape.as_list()

                self.inputs.append(op.outputs[0].name)
                self.inputs_shape.append(input_shape)
                self.inputs_dtype.append(dtype)
                self.inputs_maxval.append(op.get_attr('dtype').max)
                self.inputs_minval.append(op.get_attr('dtype').min)

                self.layers.append(make_caffe_input_layer(op.outputs[0].name, input_shape, len(self.inputs), self.param))

        print('Tensorflow GraphDef Input size: (graph version=%d)' %self.graph.version)
        for i, shape in enumerate(self.inputs_shape):
            print(self.inputs[i], shape_map_nchw2nhwc(shape))

        # Check Input Shape
        for input_shape in self.inputs_shape:
            if None in input_shape:
                sys.exit('Error: Dynamic Model input detected, Please Use -input_shape to overwrite input shape.')

        if len(self.inputs_shape) == 0:
            sys.exit('Error: Dynamic Model input detected, Please Use -input_shape to overwrite input shape.')


    def parse(self):
        logger.debug('Parsing the TensorFlow Model...')

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(self.graph, name='')

        operations = graph.get_operations()

        self.parse_input(operations)

        operations = self.preprocess(operations)

        for index, op in enumerate(operations):
            if op.type == 'Const':
                self.constant[op.outputs[0].name] = tf.get_static_value(op.outputs[0])
            elif op.type in ('Identity', 'IdentityN', 'Complex', 'FakeQuantWithMinMaxVars'):
                if tf.get_static_value(op.outputs[0]) is None:
                    self.indentity[op.outputs[0].name] = self.indentity.get(op.inputs[0].name, op.inputs[0].name)
                else:
                    self.constant[op.outputs[0].name] = tf.get_static_value(op.outputs[0])
            elif op.type in ['NoOp', 'Assert', 'Placeholder']: #ignore_op
                pass
            else:
                self.operations.append(op)

        if len(self.operations) == 0:
            sys.exit('Error: Model file is not Tensorflow Model.\n')

        # Parse all operations
        for index, tf_op in enumerate(self.operations):

            if tf_op.type not in OpMap:
                self.unsupport.append(tf_op.type)
                continue

            op = OpMap[tf_op.type](self, tf_op, index)
            op.parse()

            if op.status.parsed:
                self.operators.append(op)

        if len(self.unsupport) > 0:
            errorMsg = 'Error: Operator ' + str(list(set(self.unsupport))) + ' does not Support.\n'
            sys.exit(errorMsg)

        self.setParsed()


    def convert(self):
        logger.debug('Converting the Model...')

        for op in self.operators:
            logger.debug(op)
            for layer in op.convert():
                self.layers.append(layer)

        self.setConverted()


    def forward(self, inputs_tensor, inputs_name, outputs_name=None):
        if outputs_name[0].find('split') >= 0:
            return None

        def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
            def _imports_graph_def():
                tf.compat.v1.import_graph_def(graph_def, name="")

            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph

            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = wrap_frozen_graph(graph_def=self.graph, inputs=inputs_name, outputs=outputs_name, print_graph=True)

        outputs = frozen_func(tf.constant((inputs_tensor[0])))

        for index, output in enumerate(outputs):
            if output.dtype.is_numpy_compatible:
                outputs[index] = output.numpy().transpose(0, 3, 1, 2) if len(output.numpy().shape) == 4 and self.param['layout'] == 'NHWC' else output.numpy()
            else:
                outputs[index] = None

        return outputs[0]


    def save(self, caffe_model_path):
        save_caffe_model(caffe_model_path, self.layers)
