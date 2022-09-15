from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Split(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Split')
        self.setInited()


    def parse(self):
        self.type = 'Slice'
        super().__parse__()

        # Attributes
        self.slice_param = dict()

        # Axis
        self.slice_param['axis'] = self.attrs['axis']

        # Slice Point
        slice_point = self.attrs['split']
        for i in range(len(self.attrs['split'])):
            if i != 0:
                slice_point[i] = slice_point[i-1] + self.attrs['split'][i]
        if slice_point[-1] == self.inputs_shape[0][self.attrs['axis']]:
            slice_point.pop()
        self.slice_param['slice_point'] = slice_point

        self.attrs = self.slice_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]

