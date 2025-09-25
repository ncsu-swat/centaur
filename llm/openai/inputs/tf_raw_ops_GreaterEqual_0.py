
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_greater_equal_inputs():
    list_of_inputs = []
    x = tf.constant([5, 4, 6, 7], dtype=tf.float32)
    y = tf.constant([5, 2, 5, 10], dtype=tf.float32)
    input_dict = {'name': 'greater_equal_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant([5, 4, 6, 7], dtype=tf.int32)
    y = tf.constant([5], dtype=tf.int32)
    input_dict = {'name': 'greater_equal_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant([1, 2, 3, 4], dtype=tf.uint8)
    y = tf.constant([2, 1, 4, 3], dtype=tf.uint8)
    input_dict = {'name': 'greater_equal_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GreaterEqual"] = tf_raw_ops_greater_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GreaterEqual' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GreaterEqual'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.GreaterEqual', generated_inputs['tf.raw_ops.GreaterEqual'], lib="tf", suffix=0)
