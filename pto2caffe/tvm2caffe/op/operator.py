import re
from base_Operator import BaseOperator
from tvm2caffe.relay import get_relay_type
from util import shape_map_nhwc2nchw

from caffe_transform import caffe_layer


class Operator(BaseOperator):

    def __init__(self, model, relay, index):
        super().__init__(model, None, index)
        self.relay = relay
        self.operator_code = get_relay_type(relay)


    def __parseInput__(self):
        self.inputs = self.relay_inputs = re.compile(r'%(.+?)[,|\)|\.]').findall(self.relay.split(' = ')[-1])

        inputs = list()
        for index, input_name in enumerate(self.inputs):
            inputs = inputs + self.model.indentity.get(input_name, [input_name])
        self.inputs = inputs

        for index, input_name in enumerate(self.inputs):
            self.inputs_buf.append(self.model.constant.get(input_name, None))
            self.inputs_shape.append(shape_map_nhwc2nchw(self.model.tensor_shape.get(input_name, None)) if self.layout == 'NHWC' else self.model.tensor_shape.get(input_name, None))

        operands = re.compile(r'[\$| ](.+?) \/\* ty=').findall(self.relay.split(self.operator_code)[1].split(') /*')[0])
        for operand in operands:
            operand = operand.strip()
            if '$meta' in operand:
                self.inputs.append(operand[5:])
                self.inputs_buf.append(self.model.constant.get(operand[5:], None))
                self.inputs_shape.append(self.model.tensor_shape.get(operand[5:], None))
            elif 'meta' in operand:
                self.inputs.append(operand[4:])
                self.inputs_buf.append(self.model.constant.get(operand[4:], None))
                self.inputs_shape.append(self.model.tensor_shape.get(operand[4:], None))
            else:
                self.inputs.append('operand'+str(len(self.inputs)+1))
                self.inputs_buf.append(eval(operand))
                self.inputs_shape.append([])


    def __parseOutput__(self):
        self.outputs = [str(self.index)] if self.relay.strip().startswith(self.operator_code) else re.compile(r'%(.+?) ').findall(self.relay.split('= ')[0])

        outputs = self.outputs
        for index, output_name in enumerate(self.outputs):
            if output_name in self.model.indentity.keys():
                outputs = outputs[:outputs.index(output_name)] + self.model.indentity[output_name] + outputs[outputs.index(output_name)+1:]
        self.outputs = outputs

        for index, output_name in enumerate(self.outputs):
            self.outputs_shape.append(shape_map_nhwc2nchw(self.model.tensor_shape[output_name]) if self.layout == 'NHWC' else self.model.tensor_shape[output_name])


    def __parseAttributes__(self):
        keys = re.compile(r', ([a-zA-Z_]+)=').findall(self.relay.split(' = ')[-1].split(') /*')[0])
        for key in keys:
            if self.relay.split(key+'=', 1)[1].startswith('[['):
                self.attrs[key] = eval(re.compile(r'\[\[.+?\]\]').findall(self.relay.split(key+'=', 1)[1])[0])
            elif self.relay.split(key+'=', 1)[1].startswith('['):
                self.attrs[key] = eval(re.compile(r'[\[][0-9, -]*\]').findall(self.relay.split(key+'=', 1)[1])[0])
            elif re.compile(r'[\d-]').match(self.relay.split(key+'=', 1)[1]) is not None:
                self.attrs[key] = eval(re.compile(r'[\.\d-]+').match(self.relay.split(key+'=', 1)[1]).group(0))
            elif self.relay.split(key+'=', 1)[1].startswith('""'):
                self.attrs[key] = ''
            elif self.relay.split(key+'=', 1)[1].startswith('"'):
                self.attrs[key] = eval(re.compile(r'"(.+?)"').match(self.relay.split(key+'=', 1)[1]).group(0))
            elif re.compile(r'[A-Za-z]').match(self.relay.split(key+'=', 1)[1]) is not None:
                self.attrs[key] = eval(re.compile(r'[A-Za-z]+').match(self.relay.split(key+'=', 1)[1]).group(0))
            else:
                raise NotImplementedError


    def __parse__(self):
        self.__parseInput__()
        self.__parseOutput__()
        self.__parseAttributes__()
        self.layout = self.attrs.get('layout', self.layout)

        if len(self.inputs_buf) > 0 and self.inputs_buf[0] is not None:
            print('Constant Op Need Handle:'+str(self.operator_code)+ ' ' + self.outputs[0])

        if self.model.debug_outputs[self.operator_code] is None:
            self.model.debug_outputs[self.operator_code] = list()
        self.model.debug_outputs[self.operator_code].extend(self.outputs)


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

        self.model.indentity[self.outputs[0]] = list()
        for index, input_name in enumerate(self.inputs):
            if self.inputs_buf[index] is None:
                self.model.indentity[self.outputs[0]].append(self.model.indentity.get(input_name, input_name))
        # Handle Legacy Pad for Ignore Op
        if self.inputs[0] in self.model.pad.keys():
            self.model.pad[self.outputs[0]] = self.model.pad[self.inputs[0]]


    def unSupported(self, errorMsg=None):
        if errorMsg is not None:
            self.model.errorMsg.append('Error: Op (' + self.operator_code + '): ' + errorMsg)
        self.model.unsupport.append(self.operator_code)
