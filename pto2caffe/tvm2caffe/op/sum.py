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
        elif self.attrs['axis'] == [1] and len(self.inputs_shape[0]) == 4:
            if self.attrs['keepdims']:
                self.type = 'Permute+Reduction+Reshape+Permute'
                self.inter_blob0 = 'reduction_permute0_split'+str(self.index)
                self.inter_blob1 = 'reduction_reshape1_split'+str(self.index)
                self.inter_blob2 = 'reduction_permute2_split'+str(self.index)
                self.permute_param0 = dict(order=[0,2,3,1])
                self.reshape_param = dict(shape=dict(dim=[self.outputs_shape[0][0], self.outputs_shape[0][2], self.outputs_shape[0][3], 1] ))
                self.permute_param1 = dict(order=[0,3,1,2])
            else:
                self.type = 'Permute+Reduction'
                self.inter_blob = 'reduction_permute_split'+str(self.index)


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
        elif self.type == 'Permute+Reduction+Reshape+Permute':
            layers.append(caffe_layer(self.layer_type[0]), self.name[0], [self.inputs[0]], self.inputs_buf, [self.inter_blob0], permute_param=self.permute_param0)
            layers.append(caffe_layer(self.layer_type[1]), self.name[1], [self.inter_blob0], self.inputs_buf, [self.inter_blob1], reduction_param=self.reduction_param)
            layers.append(caffe_layer(self.layer_type[2]), self.name[2], [self.inter_blob1], self.inputs_buf, [self.inter_blob2], reshape_param=self.reshape_param)
            layers.append(caffe_layer(self.layer_type[3]), self.name[3], [self.inter_blob2], self.inputs_buf, self.outputs, permute_param=self.permute_param1)

        self.setConverted()

        return layers
