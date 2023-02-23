import sys
import torch
from compare import compare
from preprocess import get_input_tensor
from pytorch2caffe.model import Model


def check_model_file(pytorch_file):
    try:
        model = torch.jit.load(pytorch_file)
    except:
        sys.exit('Error: Model file is not Torchscript.\n')


def convert(pytorch_file, caffe_model_path, param=None):
    check_model_file(pytorch_file)

    model = Model(pytorch_file, param)
    model.parse()
    model.convert()
    caffe_net =model.save(caffe_model_path)

    input_tensor = get_input_tensor(param, param['input_shape'], param['dtype'], None)

    inputs_tensor = list()
    for index, input_name in enumerate(model.inputs):
        inputs_tensor.append(get_input_tensor(param, model.inputs_shape[index], model.inputs_dtype[index], None))

    compare(model, caffe_net, inputs_tensor, param.get('compare', -1))
