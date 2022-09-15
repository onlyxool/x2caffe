from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Log(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Log')
        self.setInited()


    def parse(self):
        self.type = 'Log'
        super().__parse__()

        # Attributes
        # Leave all arguments in LogParameter as default
        # base = -1.0 (base = e)
        # scale = 1.0
        # shift = 0.0

        self.log_param = dict()
        self.attrs = self.log_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, log_param=self.log_param)

        self.setConverted()

        return [layer]

