import tflite
import logging
import numpy as np
from base import Base
from util import shape_map_nhwc2nchw

logger = logging.getLogger('tflite2caffe')


class Operator(Base):

    def __init__(self, model, tf_op:tflite.Operator, tf_op_name:str, index:int):
        super().__init__(model, model.graph, index)
        self.op = tf_op
        self.operator_code = tf_op_name
        self.layer_type = str()
        self.inputs = list()
        self.inputs_shape = list()
        self.inputs_buf = list()
        self.layout = model.layout
        self.outputs = list()
        self.outputs_shape = list()
        self.pre = list()  # ops that before this op which to enable TFLite op
        self.post = list()  # ops that after this op which to enable TFLite op
        self.attrs = dict()


    @property
    def type(self):
        return self.layer_type if self.layer_type is not None else self.operator_code


    @property
    def name(self):
        return self.type + str(self.index)


    @property
    def shorty(self):
        return '[%s](%s)' % (self.name, self.type)


    def str(self):
        return '[' + self.name + ']  (' + self.type + ')'


    @property
    def attrs2str(self):
        attrstr = ''
        for key, value in self.attrs.items():
            attrstr = attrstr + '    ' + str(key) + ': ' + str(value) + '\n'
        return attrstr


    def __str__(self):
        inames = str([t for t in self.inputs])
        onames = str([t for t in self.outputs])
        return '\n%s\n%s    %s -> %s' % (self.shorty, self.attrs2str, inames, onames)


    def __parseInput__(self):
        for i in range(self.op.InputsLength()):
            if self.op.Inputs(i) >= 0:
                self.inputs.append(self.model.indentity.get(self.op.Inputs(i), self.op.Inputs(i)))
                self.inputs_shape.append(shape_map_nhwc2nchw(self.graph.Tensors(self.inputs[i]).ShapeAsNumpy().tolist()))
                self.inputs_buf.append(self.model.constant[self.op.Inputs(i)])
            else:
                self.inputs_buf.append(None)


    def __parseOutput__(self):
        for i in range(self.op.OutputsLength()):
            if self.op.Outputs(i) >= 0:
                self.outputs.append(self.op.Outputs(i))
                self.outputs_shape.append(shape_map_nhwc2nchw(self.graph.Tensors(self.outputs[0]).ShapeAsNumpy().tolist()))


    def parseInputOutput(self):
        self.__parseInput__()
        self.__parseOutput__()
