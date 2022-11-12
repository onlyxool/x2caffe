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
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.attrs = self.reshape_param
            self.setParsed()
        elif self.operator_code == 'squeeze':
            self.type = 'Reshape'
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.attrs = self.reshape_param
            self.setParsed()

        if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4 and len(self.inputs_shape[0]) > len(self.outputs_shape[0]):
            self.type = 'Permute+Reshape'
            self.permute = 'Reshape_' + self.name[0] + '_split' + str(self.index)
            self.permute_param = dict(order=[0,2,3,1])


    def convert(self):
        layers = list()
        if self.type == 'Permute+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[0]], [None], [self.permute], permute_param=self.permute_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.permute], self.inputs_buf, self.outputs, reshape_param=self.reshape_param))
        elif self.type == 'Reshape':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param))

        self.setConverted()

        return layers
