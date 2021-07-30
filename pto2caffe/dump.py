import os
import numpy as np
from tflite2caffe.tflite_run import get_output as tflite_get_output
from onnx2caffe.onnx_run import get_output as onnx_get_output


class Dump(object):
    def __init__(self, platform, model, model_name, input_tensor, param, dump_level=-1):
        self.platform = platform
        self.model = model
        self.dump_path = os.environ.get('MCHOME', os.environ['PWD']) + '/dump/' + model_name +'/'+ self.platform +'/'+ param['input_file']
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        self.input_tensor = input_tensor
        self.dump_level = dump_level
        self.layer_no = 0


    def operator(self, op):
        if self.dump_level >= 2:
            for index, input in enumerate(op.inputs_buf):
                if input is not None:
                    self.blob(input, op.name, op.inputs[index], index, 'i')
                else:
                    self.get_input(op, index)

        if self.dump_level >= 1:
            for index, output in enumerate(op.outputs):
                output_tensors = self.get_output(op.outputs[index])
                if isinstance(output_tensors, list): 
                    for output_tensor in output_tensors:
                        self.blob(output_tensor, op.name, op.outputs[index], index, 'o')
                else:
                    self.blob(output_tensors, op.name, op.outputs[index], index, 'o')


        self.layer_no += 1


    def blob(self, blob, layer_name, blob_name, index, io):
        file_size = str(blob.size)
        file_name = self.dump_path + '/l' + '%03d'%self.layer_no + io + str(index) + '_' + layer_name + '_'+ str(blob_name) +'_'+ file_size + '.txt'
        np.savetxt(file_name, blob.reshape(-1), fmt='%-10.6f')


    def get_output(self, blob_name):
        if self.platform == 'tflite':
            return tflite_get_output(self.model, self.input_tensor, blob_name)
        elif self.platform == 'onnx':
            return onnx_get_output(self.model, self.input_tensor, blob_name)
        else:
            pass


    def get_input(self, op, index):
        pass
