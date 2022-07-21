from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class Split(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Split')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.attrs['num_split'] == 1:
            self.model.indentity[self.op.outputs[0].name] = self.model.indentity.get(self.op.inputs[1].name, self.op.inputs[1].name)
        else:
            self.layer_type = 'Slice'

            self.slice_param = dict()
            self.slice_param['axis'] = dim_map_nhwc2nchw[int(self.inputs_buf[0])]
            slice_points = list()
            for i in range(self.attrs['num_split']):
                slice_points.append(self.outputs_shape[i][self.slice_param['axis']]*i)
            del slice_points[0]
            self.slice_param['slice_point'] = slice_points

            self.attrs = self.slice_param

            self.setParsed()


    def convert(self):
        if self.type == 'Reshape':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)
        elif self.type == 'Slice':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
