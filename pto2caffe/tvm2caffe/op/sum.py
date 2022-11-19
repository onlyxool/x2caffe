from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Sum(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'sum')
        self.setInited()


    def parse(self):
        super().__parse__()

        if 'exclude' in self.attrs.keys() and self.attrs['exclude']:
            self.unSupported('Do not support attribute exclude.')
            return

        if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4:
            for index, axis in enumerate(self.attrs['axis']):
                self.attrs['axis'][index] = dim_map_nhwc2nchw[axis] 

        if self.attrs['axis'][-1] == len(self.inputs_shape[0]) - 1:
            if not self.attrs['keepdims']:
                self.type = 'Reduction'
            else:
                self.type = 'Reduction+Reshape'
                self.inter_blob = 'reduction_reshape_split'+str(self.index)

            self.reduction_param = dict()
            self.reduction_param['operation'] = 1
            self.reduction_param['axis'] = self.attrs['axis'][0]
            self.attrs = self.reduction_param
            self.setParsed()
        else:
            print(self)
            raise NotImplementedError


    def convert(self):
        layers = list()
        if self.type == 'Reduction':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reduction_param=self.reduction_param))
        elif self.type == 'Reduction+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, [self.inter_blob], reduction_param=self.reduction_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inter_blob], [None], self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))

        self.setConverted()

        return layers
