from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Squeeze(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Squeeze')
        self.setInited()


    def parse(self):
        self.type = 'Reshape'
        super().__parse__()

        if self.op.inputs[0].shape == self.op.outputs[0].shape and self.inputs_buf[0] is None:
            self.byPassOperator()
        elif self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], np.squeeze(self.inputs_buf[0], axis=self.attrs['squeeze_dims']))
        else:
            if self.outputs_shape[0] is not None:
                self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            else:
                self.unSupported('Can\'t Support Output Shape = ' + str(self.outputs_shape[0]))
                return

            self.attrs = self.reshape_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
