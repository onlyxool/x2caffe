from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Reshape(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code in ('Reshape', 'Squeeze', 'Unsqueeze'))
        self.setInited()


    def parse(self):
        super().__parse__()

        if 'shape' in self.attrs:
            output_shape = self.attrs['shape']
        elif isinstance(self.outputs_shape[0], list) and len(self.outputs_shape[0]) > 0:
            output_shape = self.outputs_shape[0]
        elif len(self.inputs_buf) >= 2 and self.inputs_buf[1] is not None:
            output_shape = self.inputs_buf[1].tolist()
        else:
            self.unSupported('Can\'t Get Output Shape in ' + self.node.name)
            return

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0].reshape(output_shape))
        elif self.inputs_shape[0] == output_shape:
            self.byPassOperator()
        else:
            self.type = 'Reshape'
            self.reshape_param = dict(shape=dict(dim=output_shape))
            self.attrs = self.reshape_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
