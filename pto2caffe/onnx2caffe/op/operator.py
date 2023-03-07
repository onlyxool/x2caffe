from onnx import numpy_helper
from base_Operator import BaseOperator

from caffe_transform import caffe_layer


class Operator(BaseOperator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, index)
        self.node = node
        self.operator_code = node.op_type


    def __parseInput__(self):
        for input in self.node.input:
            self.inputs.append(self.model.indentity.get(input, input))
            self.inputs_buf.append(self.model.constant.get(input, None))
            if input in self.model.tensor_shape:
                self.inputs_shape.append(self.model.tensor_shape[input])
            elif input in self.model.constant:
                self.inputs_shape.append(list(self.model.constant[input].shape))
            else:
                self.inputs_shape.append(None)

#        if len(self.inputs_buf) > 0 and self.inputs_buf[0] is not None:
#            print('Constant Op Need Handle:'+str(self.operator_code)+ ' ' +self.node.name)


    def __parseOutput__(self):
        for output in self.node.output:
            self.outputs.append(output)
            self.outputs_shape.append(self.model.tensor_shape.get(output, None))


    def __convertAttributeProto__(self, attr):
        if attr.HasField('f'):
            return attr.f
        elif attr.HasField('i'):
            return attr.i
        elif attr.HasField('s'):
            return attr.s
        elif attr.HasField('t'):
            return numpy_helper.to_array(attr.t)
        elif len(attr.floats):
            return list(attr.floats)
        elif len(attr.ints):
            return list(attr.ints)
        elif len(attr.strings):
            return list(attr.strings)
        else:
            raise ValueError("Unsupported ONNX attribute: {}".format(attr))


    def __parseAttributes__(self):
        for attr in self.node.attribute:
            self.attrs[attr.name] = self.__convertAttributeProto__(attr)


    def __parse__(self):
        self.__parseInput__()
        self.__parseOutput__()
        self.__parseAttributes__()


    def convert(self):
        param = {self.layer_type.lower() + '_param': getattr(self, self.layer_type.lower() + '_param')}
        self.layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, **param))

        self.setConverted()

        return self.layers


    def byPassOperator(self):
        self.type = 'ByPassOperator'
        if len(self.outputs) == 0 or len(self.inputs) == 0:
            import sys
            sys.exit('Error: Use byPassOperator() after __parse__().')

        self.model.indentity[self.outputs[0]] = self.model.indentity.get(self.inputs[0], self.inputs[0])
        # Handle Legacy Pad for Ignore Op
        if self.node.input[0] in self.model.pad.keys():
            self.model.pad[self.node.output[0]] = self.model.pad[self.node.input[0]]


    def unSupported(self, errorMsg=None):
        if errorMsg is not None:
            self.model.errorMsg.append('Error: Op ' + self.node.name + ' (' + self.operator_code + '): ' + errorMsg)
        self.model.unsupport.append(self.operator_code)
