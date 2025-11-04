
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_convert_to_tensor_inputs():
    list_of_inputs = []

    input_dict = {
        "value": [1, 2, 3],
        "dtype": np.float32,
        "dtype_hint": np.float32,
        "name": "tensor_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "dtype": tf.float32,
        "dtype_hint": tf.float32,
        "name": "tensor_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([10], dtype=np.int32),
        "dtype": np.int32,
        "dtype_hint": np.int32,
        "name": "tensor_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([[-1, 2], [3, -4]], dtype=np.int32),
        "dtype": tf.int32,
        "dtype_hint": tf.int32,
        "name": "tensor_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "dtype": np.float32,
        "dtype_hint": np.float32,
        "name": "tensor_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([1, 2, 3, 4], dtype=np.int32),
        "dtype": tf.int32,
        "dtype_hint": tf.int32,
        "name": "tensor_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
        "dtype": tf.float32,
        "dtype_hint": tf.float32,
        "name": "tensor_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": [1.1, 2.2, 3.3],
        "dtype": np.float32,
        "dtype_hint": np.float32,
        "name": "tensor_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([[True, False], [False, True]], dtype=bool),
        "dtype": tf.bool,
        "dtype_hint": tf.bool,
        "name": "tensor_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.convert_to_tensor"] = tf_convert_to_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.convert_to_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.convert_to_tensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.convert_to_tensor', generated_inputs['tf.convert_to_tensor'], lib="tf", suffix=0)
