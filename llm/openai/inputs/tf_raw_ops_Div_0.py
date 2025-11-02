
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_raw_ops_div_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    input_dict = {
        "name": "div_1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 2: int32 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 4], [6, 8]], dtype=np.int32)
    input_dict = {
        "name": "div_2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 3: float64 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)
    input_dict = {
        "name": "div_3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 4: complex64 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    y = np.array([[2+4j, 6+8j], [10+12j, 14+16j]], dtype=np.complex64)
    input_dict = {
        "name": "div_4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 5: uint8 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[2, 4], [6, 8]], dtype=np.uint8)
    input_dict = {
        "name": "div_5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 6: int16 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[2, 4], [6, 8]], dtype=np.int16)
    input_dict = {
        "name": "div_6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 7: int64 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[2, 4], [6, 8]], dtype=np.int64)
    input_dict = {
        "name": "div_7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 8: float32 scalar tensor
    x = np.array(10.0, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    input_dict = {
        "name": "div_8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 9: float64 scalar tensor
    x = np.array(10.0, dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {
        "name": "div_9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 10: complex128 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    y = np.array([[2+4j, 6+8j], [10+12j, 14+16j]], dtype=np.complex128)
    input_dict = {
        "name": "div_10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.Div"] = tf_raw_ops_div_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Div'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Div', generated_inputs['tf.raw_ops.Div'], lib="tf", suffix=0)
