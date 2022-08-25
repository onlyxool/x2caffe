import tflite
from compare import compare2
from preprocess import get_input_tensor
from tflite2caffe.model import Model


def convert(tf_file, caffe_model_path, param=None):
    with open(tf_file, 'rb') as f:
        model_byte = f.read()
        tfmodel = tflite.Model.GetRootAsModel(model_byte, 0)
    assert(tfmodel.Version() == 3)

    model = Model(tfmodel, param, model_byte)
    model.parse()
    model.convert()
    caffe_net = model.save(caffe_model_path)

    input_tensor = get_input_tensor(param, model.inputs_shape[0], model.inputs_dtype[0], quantization_parameter=model.inputs_quantization_parameter[0])

    compare2(model, caffe_net, input_tensor, param.get('compare', -1))
