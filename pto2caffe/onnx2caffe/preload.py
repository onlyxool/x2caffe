import onnx
from onnx import shape_inference



def get_input_shape(model_path):
    onnx_model = onnx.load(model_path)
    onnx_model = shape_inference.infer_shapes(onnx_model)

    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)

    input_tensor = []
    for tensor in onnx_model.graph.initializer:
        input_tensor.append(tensor.name)
    for tensor in onnx_model.graph.sparse_initializer:
        input_tensor.append(tensor.name)

    input_shape = []
    for input in onnx_model.graph.input:
        if input.name in input_tensor:
            continue
        shape = []
        for dim in input.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        input_shape.append(shape)

    return input_shape
