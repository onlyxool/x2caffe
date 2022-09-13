from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Reshape(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code in ('Reshape', 'Squeeze', 'Unsqueeze'))
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0].reshape(self.outputs_shape[0]))
        elif self.inputs_shape[0] == self.outputs_shape[0]:
            self.byPassOperator()
        else:
            self.layer_type = 'Reshape'

            if 'shape' in self.attrs:
                self.reshape_param = dict(shape=dict(dim=self.attrs['shape']))
            elif len(self.outputs_shape[0]) > 0:
                self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            elif len(self.inputs_buf) >= 2 and self.inputs_buf[1] is not None:
                self.reshape_param = dict(shape=dict(dim=self.inputs_buf[1].tolist()))
            else:
                import sys
                sys.exit('Can\'t Get Output Shape in ' + self.node.name)

            self.attrs = self.reshape_param
            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
