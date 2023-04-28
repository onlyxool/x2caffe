from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Constantchunk(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'constantchunk')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        for index, output_shape in enumerate(self.outputs_shape):
            self.outputs_shape[index][self.slice_param['axis']] = self.slice_param['slice_point'][0]
            self.model.tensor_shape[self.outputs[index]] = self.outputs_shape[index]


    def parse(self):
        self.type = 'Slice'
        super().__parse__()

        self.slice_param = dict()
        self.slice_param['axis'] = self.attrs['dim']

        def chunk_split(length, n):
            chunk_size = length // n
            remainder = length % n
            split_points = [chunk_size * i for i in range(1,n)]

            if remainder > 0:
                split_points.append(length - remainder)

            return split_points

        self.slice_param['slice_point'] = chunk_split(self.inputs_shape[0][self.attrs['dim']], self.attrs['chunks'])

        self.attrs = self.slice_param
        self.compute_output_shape()
        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
