import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Reshape(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.reshape_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Reshape'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()


        # Option
        self.parseAttributes()
        if self.op_code == 'Unsqueeze':
            print(self.attrs)
            print(len(self.inputs_buf))
#print((self.inputs_shape), '--')
#print(self.outputs_shape)
#        out_shape = dict()
#        for i in range(self.outputs_shape[0])
#            out_shape['dim'] = 
        self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

        self.attrs = self.reshape_param
        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)
        self.setConverted()
        return [layer]
