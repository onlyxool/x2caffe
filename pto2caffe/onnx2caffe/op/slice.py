import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Cut(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.cut_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Cut'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()

        if self.inputs_buf[1].size <= 2: # Axes
            axis = list(self.inputs_buf[3])
            if min(axis) < 2 or max(axis) > 3: # Only support width and height
                raise NotImplementedError
        else:
            raise NotImplementedError

        starts = list(self.inputs_buf[1])
        ends = list(self.inputs_buf[2])
        steps = list(self.inputs_buf[4])

        self.cut_param['axis'] = axis[0]
        self.cut_param['offset'] = starts[0]
        self.cut_param['height'] = steps[0]

        if len(axis) > 1:
            self.cut_param1 = dict()
            self.cut_param1['axis'] = axis[1]
            self.cut_param1['offset'] = starts[1]
            self.cut_param1['width'] = steps[1]

        self.attrs = self.cut_param

        self.setParsed()


    def convert(self):
        layers = []
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, cut_param=self.cut_param)
        layers.append(layer)
        if hasattr(self,'cut_param1'):
            layer1 = caffe_layer(self.type, self.name, self.outputs, self.inputs_buf, self.outputs, cut_param=self.cut_param1)
            layers.append(layer1)

        self.setConverted()

        return layers

