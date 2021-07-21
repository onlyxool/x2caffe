import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator
from tflite2caffe.op.pad import computePaddingSize

logger = logging.getLogger('tflite2caffe')

class Pooling(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.MEAN: 'ReduceMean',
        tflite.BuiltinOperator.AVERAGE_POOL_2D: 'AveragePool',
        tflite.BuiltinOperator.MAX_POOL_2D: 'MaxPool',
    }


    def __init__(self, tfmodel, tfgraph, tf_op, tf_op_code, index, legacys):
        super().__init__(tfmodel, tfgraph, tf_op, tf_op_code, index, legacys)
        self.pooling_param= dict()
        self.attrs = self.pooling_param
        self.setInited()

    @property
    def type(self):
        return 'Pooling'
        
    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code in self.TypeMapping)        
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # options
        op_opt = self.op.BuiltinOptions()
        if self.op_code == tflite.BuiltinOperator.MEAN:
            opt = tflite.ReducerOptions()
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
            self.pooling_param['stride'] = 1
#            self.pooling_param['stride_h'] = 1
#            self.pooling_param['stride_w'] = 1
            self.pooling_param['ceil_mode'] = False
        else:
            opt = tflite.Pool2DOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            raise NotImplementedError

        if hasattr(opt, 'FusedActivationFunction'):
            activ_type_code = opt.FusedActivationFunction()
            if activ_type_code is not tflite.ActivationFunctionType.NONE:
                print(__file__, 'TODO: FusedActivationFunction:', activ_type_code)

        self.setParsed()

    def propagatableTensors(self):
        pass

    def transform(self):
        pass

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()
        return layer
