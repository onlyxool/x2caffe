from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Unsqueeze(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        super().__parse__()

        # Attributes
        if self.inputs_buf[0] is not None:
            self.layer_type = 'Unsqueeze'
            self.model.input_tensor[self.outputs[0]] = self.inputs_buf[0].reshape(self.outputs_shape[0])
        else:
            self.layer_type = 'Reshape'
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.attrs = self.reshape_param
            self.setParsed()


    def convert(self):
        if self.type == 'Reshape':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)
            self.setConverted()
            return [layer]
