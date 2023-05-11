from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Slice(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'slice')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        self.type = 'Slice'
        super().__parse__()

        def chunk_split(length, n):
            chunk_size = length // n
            remainder = length % n
            split_points = [chunk_size * i for i in range(1,n)]

            if remainder > 0:
                split_points.append(length - remainder)

            return split_points

        if self.inputs_buf[2] == 0 and self.inputs_buf[3] == 9223372036854775807:
            self.byPassOperator()
            return

        if self.inputs_buf[3] == 9223372036854775807:
            self.inputs_buf[3] = self.inputs_shape[0][self.inputs_buf[1]]

        if len(self.inputs_buf) == 5:
            if self.inputs_buf[2] == 0: # Start
                slice_point = [self.inputs_buf[3]]
                self.outputs.append('intermediate_' + str(self.index))
                self.outputs_shape[0][self.inputs_buf[1]] = slice_point[0]
            elif self.inputs_buf[3] >= self.inputs_shape[0][self.inputs_buf[1]]: # End
                slice_point = [self.inputs_buf[2]]
                self.outputs.insert(0, 'intermediate_' + str(self.index))
                self.outputs_shape[0][self.inputs_buf[1]] = self.outputs_shape[0][self.inputs_buf[1]] - slice_point[0]
            else:
                slice_point = [self.inputs_buf[2], self.inputs_buf[3]]
                self.outputs.insert(0, 'intermediate_' + str(self.index) + '_0')
                self.outputs.append('intermediate_' + str(self.index) + '_1')
                self.outputs_shape[0][self.inputs_buf[1]] = slice_point[1] - slice_point[0]

            if self.inputs_buf[4] != 1:
                self.unSupported('Can\'t Support Step == ' + str(self.inputs_buf[3]))
                return
        elif 'dim' in self.attrs and 'chunks' in self.attrs:
            slice_point = chunk_split(self.inputs_shape[0][self.attrs['dim']], self.attrs['chunks'])

        self.slice_param = dict()
        self.slice_param['axis'] = self.inputs_buf[1]
        self.slice_param['slice_point'] = slice_point

        self.attrs = self.slice_param

        self.compute_output_shape()
        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
