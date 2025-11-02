
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_pow_inputs():
    list_of_inputs = []
    
    # Input 1: Float32 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.float32)
    y = np.array([[2, 3], [4, 5]], dtype=np.float32)
    input_dict = {
        "name": "pow_1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Int32 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {
        "name": "pow_2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Float64 tensors
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {
        "name": "pow_3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Complex128 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    y = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    input_dict = {
        "name": "pow_4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Int64 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    input_dict = {
        "name": "pow_5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative values
    x = np.array([[-2, -3], [-4, -5]], dtype=np.float32)
    y = np.array([[2, 3], [4, 5]], dtype=np.float32)
    input_dict = {
        "name": "pow_6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single dimension tensor
    x = np.array([2, 3, 4], dtype=np.float32)
    y = np.array([2, 3, 4], dtype=np.float32)
    input_dict = {
        "name": "pow_7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Scalar tensors
    x = np.array(2, dtype=np.float32)
    y = np.array(3, dtype=np.float32)
    input_dict = {
        "name": "pow_8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Float16 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.float16)
    y = np.array([[2, 3], [4, 5]], dtype=np.float16)
    input_dict = {
        "name": "pow_9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Half tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.half)
    y = np.array([[2, 3], [4, 5]], dtype=np.half)
    input_dict = {
        "name": "pow_10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Pow'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Pow', generated_inputs['tf.raw_ops.Pow'], lib="tf", suffix=0)
