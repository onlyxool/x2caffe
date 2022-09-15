from base import Base
from onnx import numpy_helper

from caffe_transform import caffe_layer


class Operator(Base):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, index)
        self.node = node
        self.operator_code = node.op_type
        self.index = index
        self.attrs = dict()

        self.inputs = list()
        self.inputs_buf = list()
        self.inputs_shape = list()
        self.outputs = list()
        self.outputs_shape = list()

        self.type = None
        self.weight = None
        self.bias = None
        self.param = list()
        self.layers = list()


    @property
    def layer_type(self):
        if self.type is not None and self.type.find('+') >= 0:
            return self.type.split('+')
        elif self.type is not None:
            return self.type
        else:
            return self.operator_code


    @property
    def name(self):
        if self.type is not None and self.type.find('+') >= 0:
            return [layer_type+str(self.index)+'_'+str(index) for index, layer_type in enumerate(self.type.split('+'))]
        elif self.layer_type is not None:
            return self.type + str(self.index)


    @property
    def shorty(self):
        return '[%s](%s)' % (str(self.name), str(self.type))


    def inputBuf_byName(self, name):
        for index, input in enumerate(self.inputs):
            if input.find(name) > 0 or input.find(name.lower()) > 0 or input.find(name.upper()) > 0:
                return self.inputs_buf[index]


    def input_byName(self, name):
        for index, input in enumerate(self.inputs):
            if input.find(name) > 0 or input.find(name.lower()) > 0 or input.find(name.upper()) > 0:
                return input


    def str(self):
        return '[' + str(self.name) + ']  (' + str(self.type) + ')'


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


    def __parseInput__(self):
        for input in self.node.input:
            self.inputs.append(self.model.indentity.get(input, input))
            self.inputs_buf.append(self.model.constant.get(input, None))
            if input in self.model.shape:
                self.inputs_shape.append(self.model.shape[input])
            elif input in self.model.constant:
                self.inputs_shape.append(list(self.model.constant[input].shape))
            else:
                self.inputs_shape.append(None)

#        if len(self.inputs_buf) > 0 and self.inputs_buf[0] is not None:
#            print('Constant Op Need Handle:'+str(self.operator_code))


    def __parseOutput__(self):
        for output in self.node.output:
            self.outputs.append(output)
            self.outputs_shape.append(self.model.shape.get(output, None))


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
        if len(self.outputs) == 0 or len(self.inputs) == 0:
            import sys
            sys.exit('Error: Use byPassOperator() after parseInputOutput().')

        self.model.indentity[self.outputs[0]] = self.model.indentity.get(self.inputs[0], self.inputs[0])
        # Handle Legacy Pad for Ignore Op
        if self.node.input[0] in self.model.pad.keys():
            self.model.pad[self.node.output[0]] = self.model.pad[self.node.input[0]]


    def saveConstant(self, name, constant):
        self.model.constant[name] = constant


    def unSupported(self, errorMsg=None):
        if errorMsg is not None:
            self.model.errorMsg.append('Error: Op ' + self.node.name + ' (' + self.operator_code + '): ' + errorMsg)
        self.model.unsupport.append(self.operator_code)


    def propagatableTensors(self):
        """Get all layout propagable tensors of this operator.

        When we propagate layouts across the graph:
        1. Some operators may stop the propagation
            a) An operator assumes layouts of its tensors, `Conv` for example.
               Such operator needs to define the layouts of its tensors explicitly.
            b) An operator breaks layout semantic, `Reshape` for example.
               Tensors connected to this operator should be propagated.
               And the operator may need special handling regarding layout.
        2. Others may not - propagatable:
            a) An operator that is transparent to layout, such as Add.
               Just propagate the layouts.
            b) Layout can propagate across tensors of an operator, but the operator
               itself has attribution that is sensitive to layout.
               Operator needs special handling after propagation.
        This is defined per operator.

        To handle this, we firstly propagate layouts of tensors across the graph,
        and then update attributes of operators accordingly.
        """
        raise NotImplementedError("Method %s.propagatableTensors() must be overrided!" % self.type)


    def transform(self):
        """Transform the operator attributions w.r.t. propagated layouts.

        The attributions could be a tensor that describing layout related things.
        Operators that defined as 1.a, 1.b and 2.b in `layoutPropagatable()`
        are such cases. But not all of them need special treatment.
        For example, `Conv` doesn't need additional processing after propagation.

        This must be called after the layouts have been propagated across graph.
        """
        raise NotImplementedError("Method %s.transform() must be overrided!" % self.type)
