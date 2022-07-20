from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Squeeze(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Squeeze')
        self.setInited()


    def parse(self):
        self.layer_type = 'Reshape'
        super().__parse__()

#        self.outputs_shape[0] = list(np.delete(np.array(self.op.inputs[0].shape), self.attrs['squeeze_dims']))

        if None not in self.outputs_shape[0]:
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
        elif self.inputs_buf[0] is not None:
            self.reshape_param = dict(shape=dict(dim=list(self.inputs_buf[0])))
        else:
            import sys
            sys.exit('Error: Dynamic Model input detected, Please Use -inputs_shape overwirte input shape.')

        self.attrs = self.reshape_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
