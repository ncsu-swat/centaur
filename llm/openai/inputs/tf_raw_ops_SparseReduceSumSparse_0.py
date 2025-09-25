
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_reduce_sum_sparse_inputs():
    list_of_inputs = []

    input_dict1 = {
        "input_indices": tf.constant([[0, 0], [1, 1]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0], dtype=tf.float32),
        "input_shape": tf.constant([2, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([0], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        "input_indices": tf.constant([[0, 0], [0, 1], [1, 0]], dtype=tf.int64),
        "input_values": tf.constant([3, 4, 5], dtype=tf.int32),
        "input_shape": tf.constant([2, 3], dtype=tf.int64),
        "reduction_axes": tf.constant([1], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        "input_indices": tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0, 3.0], dtype=tf.float64),
        "input_shape": tf.constant([3, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([0, 1], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        "input_indices": tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.int64),
        "input_values": tf.constant([1, 2, 3], dtype=tf.int64),
        "input_shape": tf.constant([3, 3], dtype=tf.int64),
        "reduction_axes": tf.constant([-1], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_dict5 = {
        "input_indices": tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
        "input_shape": tf.constant([3, 3], dtype=tf.int64),
        "reduction_axes": tf.constant([0, -1], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input_dict6 = {
        "input_indices": tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0, 3.0], dtype=tf.float64),
        "input_shape": tf.constant([3, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {
        "input_indices": tf.constant([[0, 0], [0, 1]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0], dtype=tf.float32),
        "input_shape": tf.constant([2, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([0], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input_dict8 = {
        "input_indices": tf.constant([[0, 0], [1, 1]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0], dtype=tf.float32),
        "input_shape": tf.constant([2, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([1], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input_dict9 = {
        "input_indices": tf.constant([[0, 0], [0, 1], [1, 0]], dtype=tf.int64),
        "input_values": tf.constant([3, 4, 5], dtype=tf.int32),
        "input_shape": tf.constant([2, 3], dtype=tf.int64),
        "reduction_axes": tf.constant([1], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {
        "input_indices": tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
        "input_values": tf.constant([1, 2, 3], dtype=tf.int64),
        "input_shape": tf.constant([3, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([0, 1], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseReduceSumSparse"] = tf_sparse_reduce_sum_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseReduceSumSparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseReduceSumSparse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseReduceSumSparse', generated_inputs['tf.raw_ops.SparseReduceSumSparse'], lib="tf", suffix=0)
