from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Slice(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Slice')
        self.setInited()


    def parse(self):
        self.layer_type = 'Slice'
        super().__parse__()

        # num_slices, starts, ends, axes, steps
        if self.model.opset[0] < 10:
            num_slices = len(self.attrs['starts'])
            starts = self.attrs['starts']
            ends = self.attrs['ends']
            axes = self.attrs.get('axes', [None, None])
            steps = [1]*num_slices
        else:
            num_slices = len(self.inputs_buf[1])
            starts = list(self.inputs_buf[1])
            ends = list(self.inputs_buf[2])
            axes = list(self.inputs_buf[3]) if len(self.inputs_buf) >= 4 else None
            steps = list(self.inputs_buf[4]) if len(self.inputs_buf) >= 5 else [1]*num_slices

        if len(starts) > 1 or len(ends) > 1 or len(axes) > 1 or num_slices > 1:
            self.model.unsupport.append(self.operator_code)
            self.model.errorMsg.append('[' + self.node.name + ']: Operator Slice Do not support starts > 1. ' + self.node.name + '\'s starts is ' + str(starts))

        if ends[0] == 9223372036854775807: # int max
            ends[0] = self.inputs_shape[0][axes[0]]

        # Attributes
        if max(steps) == 1:
            axis_length = self.inputs_shape[0][axes[0]]
            self.slice_param = dict()
            self.slice_param['axis'] = axes[0]

            if starts[0] == 0:
                self.slice_param['slice_point'] = [ends[0]]
                self.outputs.append(self.name + 'useless')
            elif ends[0] == axis_length:
                self.slice_param['slice_point'] = [starts[0]]
                self.outputs.insert(0, self.name + 'useless')
            else:
                self.slice_param['slice_point'] = [starts[0], ends[0]]
                self.outputs.insert(0, self.name + 'useless0')
                self.outputs.append(self.name + 'useless1')
        else:
            self.model.unsupport.append(self.operator_code)
            self.model.errorMsg.append('[' + self.node.name + ']: Operator Slice Do not support step > 1. ' + self.node.name + '\'s steps is ' + str(steps))

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
