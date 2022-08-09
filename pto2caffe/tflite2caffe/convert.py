import tflite
from compare import compare
from preprocess import get_input_tensor
from tflite2caffe.model import Model


def convert(tf_file, caffe_model_path, param=None):
    with open(tf_file, 'rb') as f:
        model_byte = f.read()
        tfmodel = tflite.Model.GetRootAsModel(model_byte, 0)
    assert(tfmodel.Version() == 3)

    model = Model(tfmodel, param)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    input_tensor = get_input_tensor(param, model.inputs_shape[0], maxval=model.inputs_maxval[0], minval=model.inputs_minval[0])

    compare('tflite', model_byte, caffe_model_path, input_tensor, param.get('compare', -1))
