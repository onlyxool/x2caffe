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
        elif self.op_code == 'Softmax':
            return 'Softmax'
        elif self.op_code == 'PRelu':
            return 'PReLU'
        elif self.op_code == 'Relu':
            return 'ReLU'
        elif self.op_code == 'Clip':
            return 'ReLUX'
        else:
            raise NotImplementedError


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        if self.op_code == 'LeakyRelu':
            self.relu_param = dict()
            self.relu_param['negative_slope'] = self.attrs.get('alpha', 0)
            self.attrs = self.relu_param
        elif self.op_code == 'Sigmoid':
            self.sigmoid_param = dict()
            self.attrs = self.sigmoid_param
        elif self.op_code == 'Softmax':
            self.softmax_param = dict()
            self.softmax_param['axis'] = self.attrs.get('axis', 1)
            self.attrs = self.softmax_param
        elif self.op_code == 'Clip':
            self.relux_param = dict()
            if 'max' in self.attrs and 'min' in self.attrs:
                self.relux_param['x'] = self.attrs['max']
            else:
                self.relux_param['x'] = self.inputs_buf[2]
            self.attrs = self.relux_param
        elif self.op_code == 'PRelu':
            self.slope = self.inputs_buf[1]
            self.prelu_param = dict()
            if self.slope.shape[0] == 1:
                self.prelu_param['channel_shared'] = False
            else:
                self.prelu_param['channel_shared'] = False
            self.attrs = self.prelu_param
        elif self.op_code == 'Relu':
            self.relu_param = dict()
            self.relu_param['negative_slope'] = 0
            self.attrs = self.relu_param
        else:
            raise NotImplementedError(self.op_code)

        self.setParsed()


    def convert(self):
        if self.op_code == 'LeakyRelu':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)
        elif self.op_code == 'Sigmoid':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)
        elif self.op_code == 'Softmax':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, softmax_param=self.softmax_param)
        elif self.op_code == 'PRelu':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.slope, prelu_param=self.prelu_param)
        elif self.op_code == 'Relu':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)
        elif self.op_code == 'Clip':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)
        else:
            raise NotImplementedError

        self.setConverted()
        return [layer]
