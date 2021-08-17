import tflite
import logging
from dump import Dump

from base import Base

from tflite2caffe.op.pad import Pad
from tflite2caffe.op.split import Slice #TODO
from tflite2caffe.op.binary import Binary
from tflite2caffe.op.resize import Resize 
from tflite2caffe.op.concat import Concat
from tflite2caffe.op.reshape import Reshape
from tflite2caffe.op.pooling import Pooling
from tflite2caffe.op.softmax import Softmax
from tflite2caffe.op.conv import Convolution
from tflite2caffe.op.transpose import Permute
from tflite2caffe.op.activation import Activation
from tflite2caffe.op.fullyconnected import InnerProduct
from tflite2caffe.op.activation import handleFusedActivation



from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer


logger = logging.getLogger('TFlite2caffe')

OpMap = { 
    'PAD': Pad,
    'ADD': Binary,
    'MUL': Binary,
    'SPLIT': Slice,
    'MEAN': Pooling,
    'RESHAPE': Reshape,
    'SQUEEZE': Reshape,
    'SOFTMAX': Softmax,
    'RELU': Activation,
    'PRELU': Activation,
    'TRANSPOSE': Permute,
    'CONV_2D': Convolution,
    'MAX_POOL_2D': Pooling,
#    'STRIDED_SLICE': Slice,
    'CONCATENATION': Concat,
    'LEAKY_RELU': Activation,
    'RESIZE_BILINEAR': Resize,
    'AVERAGE_POOL_2D': Pooling,
    'FULLY_CONNECTED': InnerProduct,
    'DEPTHWISE_CONV_2D': Convolution,
    'RESIZE_NEAREST_NEIGHBOR': Resize,
#    'DIV': Binary,
#    'MIRROR_PAD': Pad,
}


class Model(Base):
    def __init__(self, model:tflite.Model, param):
        super().__init__(model, model.Subgraphs(0))
        self.version = model.Version()
        self.param = param
        self.operators = []
        self.layers = []
        self.legacys = []
        self.setInited()


    def parse(self):
        logger.debug("Parsing the TFLite Model...")

        if self.model.SubgraphsLength() > 1:
            raise ValueError('TFlite model include ' + str(self.model.SubgraphsLength()) + ' graphs.')

        print('TFlite Model Input size:', list(self.graph.Tensors(self.graph.Inputs(0)).ShapeAsNumpy()))

        for index in range(self.graph.OperatorsLength()):
            tf_op = self.graph.Operators(index)
            tf_op_code = self.model.OperatorCodes(tf_op.OpcodeIndex())
            tf_op_name = tflite.opcode2name(tf_op_code.BuiltinCode())

            op = OpMap[tf_op_name](self, tf_op, tf_op_code.BuiltinCode(), index)
            op.parse()
            if op.status.parsed:
                self.operators.append(op)
                if hasattr(op, 'activ_type_code'):
                    act_op = handleFusedActivation(op)
                    self.operators.append(act_op)
            else:
                self.legacys.append(op)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for i in range(self.graph.InputsLength()):
            self.layers.append(make_caffe_input_layer(self.graph.Inputs(i), self.param))
        for op in self.operators:
            logger.debug(op)
            layers = op.convert()
            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_name, caffe_path):
        save_caffe_model(caffe_name, caffe_path, self.layers)


    def dump(self, model_byte, model_name, input_tensor, dump_level=-1):
        dump = Dump('tflite', model_byte, model_name, input_tensor, self.param, dump_level)
        from progress_bar import ProgressBar
        progressBar = ProgressBar(len(self.operators), 0, "TFlite dump processing")
        for i, op in enumerate(self.operators):
            dump.operator(op)
            progressBar.setValue(i)
        progressBar.onCancel()
