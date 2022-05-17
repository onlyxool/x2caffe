from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Sub(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Sub')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.layer_type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
        else:
            self.layer_type = 'Scale'

            self.bias = -self.inputs_buf[1]

            self.scale_param = dict()
            self.scale_param['bias_term'] = True

            # Axis
            if self.bias.shape != () and self.bias.shape != []: 
                self.scale_param['axis'] = self.inputs_shape[0].index(self.bias.shape[0])
                self.scale_param['num_axes'] = len(self.bias.shape)
            else:
                self.scale_param['num_axes'] = 0 

            self.attrs = self.scale_param

        self.setParsed()


    def convert(self):
        if self.layer_type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        if self.layer_type == 'Scale':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, None, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
