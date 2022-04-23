import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Sub(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            # Eltwise
            self.layer_type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
        else:
            # Scale
            self.layer_type = 'Scale'
            if self.inputs_buf[0] is not None:
                bias_index = 0
                input_index = 1
            else:
                bias_index = 1
                input_index = 0

            self.bias = -self.inputs_buf[bias_index]

            self.scale_param = dict()
            self.scale_param['bias_term'] = True

            # Axis
            if self.bias.shape != () and self.bias.shape != []:
                self.scale_param['axis'] = self.inputs_shape[input_index].index(self.bias.shape[0])
                self.scale_param['num_axes'] = len(self.bias.shape)
            else:
                self.scale_param['num_axes'] = 0

            self.attrs = self.scale_param

        self.setParsed()


    def convert(self):
        if self.layer_type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.layer_type == 'Scale':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)
        else:
            raise NotImplementedError

        self.setConverted()

        return [layer]
