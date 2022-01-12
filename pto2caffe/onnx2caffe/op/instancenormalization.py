import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class InstanceNormalization(Operator):
    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.batch_norm_param = dict()
        self.scale_param = dict()
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Weight
        self.weight = self.inputs_buf[1]

        # Bias
        self.bias = self.inputs_buf[2]

        # Option
        self.parseAttributes()
        self.batch_norm_param['eps'] = self.attrs.get('epsilon', 1e-5)
        self.batch_norm_param['use_global_stats'] = False

        self.scale_param['bias_term'] = True

        self.attrs = self.batch_norm_param
        self.setParsed()


    @property
    def type(self):
        return 'BatchNorm'


    def convert(self):
        layer0 = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, batch_norm_param=self.batch_norm_param)
        layer1 = caffe_layer('Scale', 'Scale'+str(self.index), self.outputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()
        return [layer0, layer1]
