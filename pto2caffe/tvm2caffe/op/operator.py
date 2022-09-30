import re
from base import Base
from tvm2caffe.relay import get_relay_type, get_tensor_shape

from caffe_transform import caffe_layer


class Operator(Base):

    def __init__(self, model, relay, index):
        super().__init__(model, None, index)
        self.relay = relay
        self.operator_code = get_relay_type(relay)
        self.attrs = dict()

        self.inputs = list()
        self.inputs_buf = list()
        self.inputs_shape = list()
        self.outputs = list()
        self.outputs_shape = list()

        self.type = self.operator_code
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
        ishape = str(self.inputs_shape)
        oshape = str(self.outputs_shape)
        inbuf = str([None if t is None else 'np.array' for t in self.inputs_buf])
        return '\n%s\n%s    %s -> %s\n    %s -> %s\n    %s' % (self.shorty, self.attrs2str, inames, onames, ishape, oshape, inbuf)


    def __parseInput__(self):
        self.inputs = re.compile(r'%(.+?)[,|\)|\.]').findall(self.relay.split(' = ')[-1])

        constants = re.compile(r'meta(.+?) */').findall(self.relay.split(' = ')[-1].split(') /*')[0])
        if len(constants) > 0:
            index = self.relay.split(' = ')[-1].split(constants[0])[0].count('%')
        for constant in constants:
            self.inputs.insert(index, constant)

        for input_name in self.inputs:
            self.inputs_buf.append(self.model.constant.get(input_name, self.model.constant.get(input_name[1:], None)))
            self.inputs_shape.append(self.model.tensor_shape.get(input_name, self.model.tensor_shape.get(input_name[1:], None)))

#        if len(self.inputs_buf) > 0 and self.inputs_buf[0] is not None:
#            print('Constant Op Need Handle:'+str(self.operator_code)+ ' ' +self.node.name)


    def __parseOutput__(self):
        self.outputs = re.compile(r'%(.+?) ').findall(self.relay.split('= ')[0])
        for index, output_name in enumerate(self.outputs):
            shape_str = get_tensor_shape(self.relay.split(') /*')[-1])
            self.outputs_shape.append(eval('['+shape_str[index]+']') if len(shape_str) > 0 else None)
            self.model.tensor_shape[output_name] = eval('['+shape_str[index]+']') if len(shape_str) > 0 else None


    def __parseAttributes__(self):
        keys = re.compile(r', ([a-zA-Z_]+)=').findall(self.relay.split(' = ')[-1].split(') /*')[0])
        for key in keys:
            if self.relay.split(key+'=')[-1].startswith('['):
                self.attrs[key] = eval(re.compile(r'[\[][0-9, -]*\]').findall(self.relay.split(key+'=')[-1])[0])
            elif re.compile(r'[\d-]').match(self.relay.split(key+'=')[-1]) is not None:
                self.attrs[key] = eval(re.compile(r'[\d-]+').match(self.relay.split(key+'=')[-1]).group(0))
            elif self.relay.split(key+'=')[-1].startswith('"'):
                self.attrs[key] = eval(re.compile(r'"(.+?)"').match(self.relay.split(key+'=')[-1]).group(0))
            elif re.compile(r'[A-Za-z]').match(self.relay.split(key+'=')[-1]) is not None:
                self.attrs[key] = eval(re.compile(r'[A-Za-z]+').match(self.relay.split(key+'=')[-1]).group(0))
            else:
                raise NotImplementedError


    def __parse__(self):
        print(self.relay)
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
            sys.exit('Error: Use byPassOperator() after parseInputOutput().')

        self.model.indentity[self.outputs[0]] = self.model.indentity.get(self.inputs[0], self.inputs[0])
        # Handle Legacy Pad for Ignore Op
        if self.node.input[0] in self.model.pad.keys():
            self.model.pad[self.node.output[0]] = self.model.pad[self.node.input[0]]


    def saveConstant(self, name, constant):
        self.type = 'Constant'
        self.model.constant[name] = constant


    def unSupported(self, errorMsg=None):
        if errorMsg is not None:
            self.model.errorMsg.append('Error: Op ' + self.node.name + ' (' + self.operator_code + '): ' + errorMsg)
        self.model.unsupport.append(self.operator_code)
