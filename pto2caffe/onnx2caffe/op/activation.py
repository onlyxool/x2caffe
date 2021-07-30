import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Activation(Operator):
    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()

    @property
    def type(self):
        if self.op_code == 'LeakyRelu':
            return 'ReLU'
        elif self.op_code == 'Sigmoid':
            return 'Sigmoid'
        elif self.op_code == 'PRelu':
            return 'PReLU'
        elif self.op_code == 'Relu':
            return 'ReLU'
        else:
            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        if self.op_code == 'LeakyRelu':
            self.relu_param = dict()
            self.relu_param['negative_slope'] = self.attrs.get('alpha', 0)
            self.attrs = self.relu_param
        elif self.op_code == 'Sigmoid':
            self.sigmoid_param = dict()
            self.attrs = self.sigmoid_param
        elif self.op_code == 'PRelu':
            print('Prelu')
        elif self.op_code == 'Relu':
            self.relu_param = dict()
            self.relu_param['negative_slope'] = 0
            self.attrs = self.relu_param
        else:
            print('Error', self.op_code)

        self.setParsed()


    def propagatableTensors(self):
        return self.inputs + self.outputs


    def transform(self):
        pass


    def convert(self):
        if self.op_code == 'LeakyRelu':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)
        elif self.op_code == 'Sigmoid':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)
        elif self.op_code == 'PRelu':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, prelu_param=self.prelu_param)
        elif self.op_code == 'Relu':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)
        else:
            raise NotImplementedError

        self.setConverted()
        return [layer]
