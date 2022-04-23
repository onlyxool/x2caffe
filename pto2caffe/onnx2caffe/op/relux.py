import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class ReLUX(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        self.layer_type = 'ReLUX'
        super().__parse__()

        # Attributes
        self.relux_param = dict()
        if 'max' in self.attrs and 'min' in self.attrs:
            self.relux_param['x'] = self.attrs['max']
        else:
            self.relux_param['x'] = self.inputs_buf[2]
        self.attrs = self.relux_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)

        self.setConverted()

        return [layer]
