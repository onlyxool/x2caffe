import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Slice(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.slice_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Slice'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()

        if self.model.opset[0] < 10:
            num_slices = len(self.attr['starts'])
            starts = self.attr['starts']
            axes = self.attr['axes']
            ends = self.attr['ends']
            steps = [1]*num_slices
        else:
            num_slices = len(self.inputs_buf[1])
            starts = list(self.inputs_buf[1])
            ends = list(self.inputs_buf[2])
            axes = list(self.inputs_buf[3])
            steps = list(self.inputs_buf[4]) if len(self.inputs_buf) >= 5 else [1]*num_slices

        if num_slices > 1:
            raise NotImplementedError

        if max(steps) == 1:
            axis_length = self.inputs_shape[0][axes[0]]
            self.slice_param['axis'] = axes[0]

            if starts[0] == 0:
                self.slice_param['slice_point'] = [ends[0]]
                self.outputs.append(self.name + 'useless')
            elif ends[0] == axis_length:
                self.slice_param['slice_point'] = [starts[0]]
                self.outputs.insert(0, self.name + 'useless')
            else:
                self.slice_param['slice_point'] = [starts[0], ends[0]]
                self.outputs.insert(0, self.name + 'useless0')
                self.outputs.append(self.name + 'useless1')
        else:
            import sys
            errorMsg = 'Do not support step > 1. ' + self.name + '\'s steps is ' + str(steps) + '\n'
            sys.exit(errorMsg)

        self.setParsed()


    def convert(self):
        layers = []

        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)
        layers.append(layer)

        self.setConverted()

        return layers

