
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_empty_inputs():
    list_of_inputs = []
    input_dict_1 = {
        "shape": tf.constant([2, 3], dtype=tf.int32),
        "dtype": tf.float32,
        "init": True,
        "name": "empty_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {
        "shape": tf.constant([5]),
        "dtype": tf.int64,
        "init": False,
        "name": "empty_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {
        "shape": tf.constant([1, 4, 2]),
        "dtype": tf.string,
        "init": True,
        "name": "empty_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {
        "shape": tf.constant([0, 0]),
        "dtype": tf.bool,
        "init": False,
        "name": "empty_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_dict_5 = {
        "shape": tf.constant([2, 2, 3, 1]),
        "dtype": tf.float16,
        "init": True,
        "name": "empty_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_dict_6 = {
        "shape": tf.constant([10, 1]),
        "dtype": tf.complex64,
        "init": False,
        "name": "empty_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_dict_7 = {
        "shape": tf.constant([2]),
        "dtype": tf.int32,
        "init": True,
        "name": "empty_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_dict_8 = {
         "shape": tf.constant([3, 4, 5]),
        "dtype": tf.float32,
        "init": False,
        "name": "empty_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_dict_9 = {
        "shape": tf.constant([1, 5, 1, 2]),
        "dtype": tf.string,
        "init": True,
        "name": "empty_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_dict_10 = {
        "shape": tf.constant([2, 3, 0, 1]),
        "dtype": tf.bool,
        "init": False,
        "name": "empty_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    return list_of_inputs

generated_inputs["tf.raw_ops.Empty"] = tf_empty_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Empty' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Empty'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Empty', generated_inputs['tf.raw_ops.Empty'], lib="tf", suffix=0)
