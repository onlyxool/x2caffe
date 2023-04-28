from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Select(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'select')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        del self.outputs_shape[0][self.inputs_buf[1]]
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        self.type = 'Slice+Reshape'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import torch
            self.saveConstant(self.outputs[0], torch.select(torch.Tensor(self.inputs_buf[0]), self.inputs_buf[1], self.inputs_buf[2]).detach().numpy())
        else:
            self.slice_param = dict()
            self.slice_param['axis'] = self.inputs_buf[1]
            self.slice_param['slice_point'] = [self.inputs_buf[2], self.inputs_buf[2] + 1]
    
            for index, slice_point in enumerate(self.slice_param['slice_point']):
                if slice_point == 0 or slice_point == self.inputs_shape[0][self.inputs_buf[1]]:
                    del self.slice_param['slice_point'][index]
    
            self.attrs = self.slice_param
    
            self.compute_output_shape()
            self.setParsed()


    def convert(self):
        layers = list()

        if len(self.slice_param['slice_point']) == 2:
            self.inter = ['intermediate_'+str(self.index)+'_s', self.interblob[0], 'intermediate_'+str(self.index)+'_e']
        elif len(self.slice_param['slice_point']) == 1 and self.inputs_buf[2] == 0:
            self.inter = [self.interblob[0], 'intermediate_'+str(self.index)+'_e']
        elif len(self.slice_param['slice_point']) == 1 and self.inputs_buf[2] == self.inputs_shape[0][self.inputs_buf[1]] - 1:
            self.inter = ['intermediate_'+str(self.index)+'_s', self.interblob[0]]

        layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, self.inter, slice_param=self.slice_param))
        layers.append(caffe_layer(self.layer_type[1], self.name[1], self.interblob, self.inputs_buf, self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))

        self.setConverted()

        return layers
