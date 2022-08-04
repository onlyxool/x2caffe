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
            if self.inputs_buf[1].size == 1:
                begin = int(self.inputs_buf[1])
                size = int(self.inputs_buf[2])
                end = self.inputs_buf[0].shape[0] if size == -1 else (begin + size)
                self.model.constant[self.outputs[0]] = self.inputs_buf[0][begin:end]
            elif self.inputs_buf[1].size == 2:
                begins = self.inputs_buf[1]
                sizes = self.inputs_buf[2]
                ends = list()
                for i, size in enumerate(sizes):
                    ends.append(self.inputs_buf[0].shape[i] if size == -1 else (begins[i] + sizes[i]))
                self.model.constant[self.outputs[0]] = self.inputs_buf[0][begins[0]:ends[0], begins[1]:ends[1]]
            elif self.inputs_buf[1].size == 3:
                begins = self.inputs_buf[1]
                sizes= self.inputs_buf[2]
                ends = list()
                for i, size in enumerate(sizes):
                    ends.append(self.inputs_buf[0].shape[i] if size == -1 else (begins[i] + sizes[i]))
                self.model.constant[self.outputs[0]] = self.inputs_buf[0][begins[0]:ends[0], begins[1]:ends[1], begins[2]:ends[2]]
            else:
                raise NotImplementedError(self.op.name)
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
