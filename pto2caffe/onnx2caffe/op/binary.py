import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Binary(Operator):
    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()

    @property
    def type(self):
        if self.op_code == 'Add':
            return 'Eltwise'
        elif self.op_code == 'Sub':
            return 'TODO'
        elif self.op_code == 'Mul':
            return 'TODO'
        elif self.op_code == 'Pow':
            return 'TODO'
        else:
            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        if self.op_code == 'Add':
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1
            self.attrs = self.eltwise_param

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass

    def convert(self):
        if self.op_code == 'Add':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        else:
            raise NotImplementedError

        self.setConverted()
        return [layer]
