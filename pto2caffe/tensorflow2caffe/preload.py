import tensorflow as tf

def get_input_shape(model_path): 
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    fuck = graph_def.node
    for node in fuck:
        if node.op == 'Placeholder':
            print(dir(node.attr))

#    tf_ops = []
#    with tf.compat.v1.Session() as sess:
#    # Get Ops
#        sess.graph.as_default()
#        tf.compat.v1.import_graph_def(graph_def, name='')
#        tf_ops = [op for op in sess.graph.get_operations() if op.type != 'Const' and op.type != 'Identity']
#
#        # Graph Input
#        inputs_shape = []
#        for op in tf_ops:
#            if op.type == 'Placeholder':
#                input_shape = []
#                for dim in op.get_attr('shape').dim:
#                    input_shape.append(dim.size)
#                inputs_shape.append(input_shape)
##inputs.append(op.outputs[0].name)
#
#    print(inputs_shape)
