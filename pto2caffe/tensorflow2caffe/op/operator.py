from base import Base
from util import shape_map_nhwc2nchw


class Operator(Base):

    def __init__(self, model, tf_op, index):
        super().__init__(model, model.graph, index)
        self.op = tf_op
        self.operator_code = tf_op.type
        self.layer_type = None
        self.inputs = []
        self.inputs_shape = []
        self.inputs_buf = []
        self.outputs = []
        self.outputs_shape = []
        self.pre = []  # ops that before this op which to enable TensorFlow op
        self.post = []  # ops that after this op which to enable TensorFlow op
        self.attrs = dict()


    @property
    def type(self):
        return self.layer_type if self.layer_type is not None else self.operator_code


    @property
    def name(self):
        return self.type + str(self.index)


    @property
    def shorty(self):
        return '[%s](%s)' % (self.name, self.type)


    def str(self):
        return '[' + self.name + ']  (' + self.type + ')'


    @property
    def attrs2str(self):
        attrstr = ''
        for key, value in self.attrs.items():
            attrstr = attrstr + '    ' + str(key) + ': ' + str(value) + '\n'
        return attrstr


    def __str__(self):
        inames = str([t for t in self.inputs])
        onames = str([t for t in self.outputs])
        return '\n%s\n%s    %s -> %s' % (self.shorty, self.attrs2str, inames, onames)


    def debug(self):
        print('\nOp:', self.name, self.op.name, self.operator_code)

        print('Input:')
        for op_input in self.op.inputs:
            print('    ', op_input)
            print('     Buf', self.model.constant.get(self.model.indentity.get(op_input.name, op_input.name), None))

        print('Output:')
        for op_output in self.op.outputs:
            print('    ', op_output)

        print('Attrs:', self.attrs, end='\n\n')


    def ndim(self, dim):
        return self.layout.find(dim)


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
