from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Slice(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Slice')
        self.setInited()


    def parse(self):
        self.layer_type = 'Slice'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            input = tf.constant(self.inputs_buf[0], self.op.inputs[0].dtype)
            begin = tf.constant(self.inputs_buf[1], self.op.inputs[1].dtype)
            size = tf.constant(self.inputs_buf[2], self.op.inputs[2].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.Slice(input=input, begin=begin, size=size, name=None).numpy())
        else:
            raise NotImplementedError(self.op.name)
            self.slice_param = dict()
            self.slice_param['axis'] = int(axis_index[0]) if self.layout == 'NCHW' else dim_map_nhwc2nchw[int(axis_index[0])]
            self.slice_param['slice_point'] = [slice_point]

            self.attrs = self.slice_param


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
