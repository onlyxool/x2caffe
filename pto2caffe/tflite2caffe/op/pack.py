import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Pack(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'PACK')
        self.setInited()


    def parse(self):
        self.type = 'Pack'

        self.parseInputOutput()

        for input_buf in self.inputs_buf:
            if input_buf is not None:
                constant = True
            else:
                constant = False
                break

        if constant:
            self.saveConstant(self.outputs[0], np.stack(self.inputs_buf))
        else:
            self.type = 'Reshape+' * len(self.inputs) + 'Concat'

            op_opt = self.op.BuiltinOptions()
            opt = tflite.PackOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            #opt.ValuesCount()

            # Reshape Attribute
            self.reshape_param = list()
            for index, input_name in enumerate(self.inputs):
                if opt.Axis() == 0:
                    self.reshape_param.append(dict(shape=dict(dim=[1]+self.inputs_shape[index])))
                elif opt.Axis() == len(self.outputs_shape[0]) - 1:
                    self.reshape_param.append(dict(shape=dict(dim=self.inputs_shape[index]+[1])))
                else:
                    self.unSupported('Can\'t support: axis == ' + str(opt.Axis()))
                    return

            # Concat Attribute
            self.concat_param = dict()
            self.concat_param['axis'] = dim_map_nhwc2nchw[opt.Axis()] if self.layout == 'NHWC' and self.outputs_shape[0] != self.op.outputs[0].shape.as_list() else opt.Axis()

            self.attrs = self.concat_param

            self.setParsed()


    def convert(self):
        layers = list()
        for index, input_name in enumerate(self.inputs):
            layers.append(caffe_layer(self.layer_type[index], self.name[index], [self.inputs[index]], [None], self.interblob[index], reshape_param=self.reshape_param[index]))

        layers.append(caffe_layer(self.layer_type[len(self.inputs)], self.name[len(self.inputs)], self.interblob, self.inputs_buf, self.outputs, concat_param=self.concat_param))

        self.setConverted()

        return layers
