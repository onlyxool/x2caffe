import re
import sys
import tvm
import logging
from base import Base

from tvm2caffe.relay import preprocess, get_relay_type, get_tensor_shape

from tvm2caffe.op.operator import Operator
from tvm2caffe.op.convolution import Convolution

OpMap = {
    'nn.conv2d': Convolution,
}


logger = logging.getLogger('Tvm2Caffe')


import json
class Model(Base):

    def __init__(self, model, model_params, param):
        model_txt = model.astext(show_meta_data=True)
        super().__init__(model, model_txt[model_txt.find('main'):].split('}')[0])
        self.param = param
        self.model_params = model_params


        self.inputs = list()
        self.inputs_shape = list()
        self.inputs_dtype = list()
        self.inputs_maxval = list()
        self.inputs_minval = list()

        self.outputs = list()
        self.outputs_shape = list()
        self.outputs_dtype = list()
        self.outputs_maxval = list()
        self.outputs_minval = list()

        self.pad = dict()
        self.tensor_shape = dict()
        self.layers = list()
        self.constant = dict()
        self.errorMsg = list()
        self.indentity = dict()
        self.operators = list()
        self.unsupport = list()

        self.setInited()


    def get_ty_tensor(self, src):
        def remove_brackets(src):
            src = src.strip()
            if any([src.startswith(s) for s in ('(', '[', '{')]) and any([src.endswith(s) for s in (')', ']', '}')]):
                src = src[1:]
                src = src[:-1]
            return src

        def find_tensor(src):
            return re.compile(r'Tensor\[(.+?)\]').findall(src)

        ty = src.split('ty=')[-1].split('*/')[0]
        ty = remove_brackets(ty)
        return find_tensor(ty)


    def parse(self):
        logger.debug("Parsing the TVM Model...")

        # Model Params
        for key in self.model_params.keys():
            self.constant[key] = self.model_params[key].numpy()
            self.tensor_shape[key] = list(self.model_params[key].numpy().shape)

        # Meta Data
        if self.model.astext(show_meta_data=True).find('metadata') >= 0:
            metadata = tvm.ir.load_json(self.model.astext(show_meta_data=True).split('[metadata]')[-1])
            for index, meta in enumerate(metadata['relay.Constant']):
                self.constant['[relay.Constant]['+str(index)+']'] = meta.data.numpy()
                self.tensor_shape['[relay.Constant]['+str(index)+']'] = list(meta.data.numpy().shape)

        inputs_str, relays = self.graph.split('{')

        print('Tvm Model Input size:')
#        print(re.compile(r'%(.+?):').findall(inputs_str))
        for inputs in inputs_str.split('(%')[-1].split(', %'):
            if inputs.split(': ')[0] not in self.constant.keys():
                self.inputs.append(inputs.split(': ')[0])
                self.inputs_shape.append(eval('['+get_tensor_shape(inputs.split(': ')[-1])[0]+']'))
                self.inputs_dtype.append(re.compile(r'\), (.+?)\]').findall(inputs.split(': ')[-1])[0])

        for index, inputs_name in enumerate(self.inputs):
            print(inputs_name, end='')
            print(':', self.inputs_shape[index], self.inputs_dtype[index])
            self.tensor_shape[inputs_name] = self.inputs_shape[index]


        for index, relay in enumerate(relays.split(';')):
            relay = preprocess(relay)
#            relay_type = get_relay_type(relay)
#            if relay_type not in OpMap: # Unsupport OP
#                self.unsupport.append(relay_type)
#                continue

#            op = OpMap[relay_type](self, relay, index)
            op = Operator(self, relay, index)
            op.__parse__()

            if op.status.parsed:
                self.operators.append(op)
            print(op)
        sys.exit('EXIT!!!!')
        for errorMsg in list(set(self.errorMsg)):
            print(errorMsg)

        if len(self.unsupport) > 0:
            errorMsg = 'Error: Operator ' + str(list(set(self.unsupport))) + ' does not Support.\n'
            sys.exit(errorMsg)

        self.setParsed()

