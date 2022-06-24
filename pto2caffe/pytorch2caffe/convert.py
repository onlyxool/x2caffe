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

    input_tensor = get_input_tensor(param,param['input_shape'])

    model = Model(pytorch_file, param, input_tensor)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    compare('pytorch', model, caffe_model_path, input_tensor, param.get('compare', -1))
