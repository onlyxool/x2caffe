import onnx
from onnx import shape_inference
from onnx2caffe.model import Model

def convert(onnx_file, input_tensor, caffe_model_name, caffe_model_path, dump_level=-1, param=None):
    onnx_model = onnx.load(onnx_file)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    onnx_model = shape_inference.infer_shapes(onnx_model)

    model = Model(onnx_model, param)
    model.parse()
    model.convert()
    model.save(caffe_model_name, caffe_model_path)
