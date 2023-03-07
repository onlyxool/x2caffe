from util import shape_map_nhwc2nchw
from base_Operator import BaseOperator


class Operator(BaseOperator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, model.graph, index)
        self.op = tf_op
        self.operator_code = tf_op.type


    def __parseInput__(self):
        for index, op_input in enumerate(self.op.inputs):
            self.inputs.append(self.model.indentity.get(op_input.name, op_input.name))
            self.inputs_buf.append(self.model.constant.get(self.inputs[index], None))
            if op_input.shape.is_fully_defined():
                self.inputs_shape.append(shape_map_nhwc2nchw(op_input.shape.as_list()) if self.layout == 'NHWC' else op_input.shape.as_list())
            else:
                self.inputs_shape.append(None)


    def __parseOutput__(self):
        for index, op_output in enumerate(self.op.outputs):
            self.outputs.append(op_output.name)
            if op_output.shape.is_fully_defined():
                self.outputs_shape.append(shape_map_nhwc2nchw(op_output.shape.as_list()) if self.layout == 'NHWC' else op_output.shape.as_list())
            else:
                self.outputs_shape.append(None)


    def __convertAttributeProto__(self, attr, name):
        if attr[name].HasField('b'):
            return attr[name].b
        elif attr[name].HasField('f'):
            return attr[name].f
        elif attr[name].HasField('func'):
            return attr[name].func
        elif attr[name].HasField('i'):
            return attr[name].i
        elif attr[name].HasField('list'):
            return list(attr[name].list.i)
        elif attr[name].HasField('s'):
            return attr[name].s.decode('utf-8')
        elif attr[name].HasField('shape'):
            return attr[name].shape
        elif attr[name].HasField('tensor'):
            return attr[name].tensor
        elif attr[name].HasField('type'):
            return attr[name].type
        else:
            raise ValueError("Unsupported TensorFlow attribute: {}".format(attr))


    def __parseAttributes__(self):
        for op_attr in self.op.op_def.attr:
            self.attrs[op_attr.name] = self.__convertAttributeProto__(self.op.node_def.attr, op_attr.name)
        self.layout = self.attrs.get('data_format', self.model.layout)


    def __parse__(self):
        self.__parseAttributes__()
        self.__parseInput__()
        self.__parseOutput__()


    def byPassOperator(self):
        self.type = 'ByPassOperator'
        if len(self.outputs) == 0 or len(self.inputs) == 0:
            import sys
            sys.exit('Error: Use byPassOperator() after __parse__().')

        self.model.indentity[self.outputs[0]] = self.model.indentity.get(self.inputs[0], self.inputs[0])
        # Handle Legacy Pad for Ignore Op
        if self.inputs[0] in self.model.pad.keys():
            self.model.pad[self.outputs[0]] = self.model.pad[self.inputs[0]]


    def unSupported(self, errorMsg=None):
        if errorMsg is not None:
            self.model.errorMsg.append('Error: Op ' + self.op.name + ' (' + self.operator_code + '): ' + errorMsg)
        self.model.unsupport.append(self.operator_code)
