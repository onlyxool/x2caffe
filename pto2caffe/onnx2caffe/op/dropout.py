from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Dropout(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Dropout')
        self.setInited()


    def parse(self):
        self.layer_type = 'Dropout'
        super().__parse__()

        if len(self.outputs) == 2: # Remove output mask
            self.outputs.pop()
            self.outputs_shape.pop()

        # Attributes
        self.dropout_param = dict()
        self.dropout_param['dropout_ratio'] = self.attrs['ratio']
        self.attrs = self.dropout_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, dropout_param=self.dropout_param)

        self.setConverted()

        return [layer]

