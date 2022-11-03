from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Reshape(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code in ('reshape', 'squeeze'))
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0].reshape(self.attrs['newshape']))
        elif self.inputs_shape[0] is not None and self.outputs_shape[0] is not None and self.inputs_shape[0] == self.outputs_shape[0]:
            self.byPassOperator()
        elif self.operator_code == 'reshape':
            self.type = 'Reshape'
            self.reshape_param = dict(shape=dict(dim=self.attrs['newshape']))
            self.attrs = self.reshape_param
            self.setParsed()
        elif self.operator_code == 'squeeze':
            self.type = 'Reshape'
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.attrs = self.reshape_param
            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
