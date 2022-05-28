import tflite
from compare import compare
from preprocess import preprocess
from tflite2caffe.model import Model


def convert(tf_file, input_tensor, caffe_model_path, dump_level=-1, param=None):
    with open(tf_file, 'rb') as f:
        model_byte = f.read()
        tfmodel = tflite.Model.GetRootAsModel(model_byte, 0)
    assert(tfmodel.Version() == 3)

    model = Model(tfmodel, param)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    input_tensor = preprocess(input_tensor, param)

    compare('tflite', model_byte, caffe_model_path, input_tensor, param.get('compare', -1))
