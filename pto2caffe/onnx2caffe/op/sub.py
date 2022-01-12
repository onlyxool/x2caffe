import logging
import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')



class Sub(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        if hasattr(self, 'eltwise_param'):
            return 'Eltwise'
        elif hasattr(self, 'scale_param'):
            return 'Scale'
        else:
            return self.op_code


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()

        for legacy in self.model.legacys:
            for i, input in enumerate(self.inputs):
                if input == legacy.outputs[0] and legacy.type == 'Constant':
                    self.inputs_buf[i] = legacy.inputs_buf[0]

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
        else:
            if self.inputs_buf[0] is not None:
                bias_index = 0
                input_index = 1
            else:
                bias_index = 1
                input_index = 0

            if self.inputs_buf[1].shape == () or self.inputs_buf[1].shape == []:
                self.inputs_buf[1] = np.ones(self.inputs_shape[0]) * self.inputs_buf[1]
                self.inputs_shape[1] = self.inputs_shape[0]

            self.bias = -self.inputs_buf[bias_index]
            self.weight = np.ones(self.bias.shape, dtype=None, order='C')

            # Attribute
            self.scale_param = dict()

            if self.model.opset[0] >= 7:
                self.scale_param['axis'] = self.inputs_shape[input_index].index(self.bias.shape[0])
            else:
                self.scale_param['axis'] = self.attrs['axis']

            self.scale_param['bias_term'] = True
            self.scale_param['num_axes'] = len(self.bias.shape)
            self.attrs = self.scale_param

        self.setParsed()


    def convert(self):
        if hasattr(self, 'eltwise_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif hasattr(self, 'scale_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
