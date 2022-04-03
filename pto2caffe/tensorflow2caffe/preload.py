import tensorflow as tf

def get_input_shape(model_path): 
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    inputs_shape = []
    for node in graph_def.node:
        if node.op == 'Placeholder':
            input_shape = []
            for dim in node.attr['shape'].shape.dim:
                input_shape.append(dim.size if dim.size != -1 else 1)
            inputs_shape.append(input_shape)

    return inputs_shape
