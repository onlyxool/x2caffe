import tflite
from compare import compare
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

    inputs_tensor = list()
    for index, input_name in enumerate(model.inputs):
        inputs_tensor.append(get_input_tensor(param, model.inputs_shape[index], model.inputs_dtype[index], quantization_parameter=model.inputs_quantization_parameter[index]))

    compare(model, caffe_net, inputs_tensor, param.get('compare', -1))
