
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_convert_to_tensor_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test1"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    value = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "test2"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    value = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    dtype = tf.float64
    dtype_hint = tf.float64
    name = "test3"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test4"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    dtype = tf.int64
    dtype_hint = tf.int64
    name = "test5"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    value = np.array([1.0, 2.0, None, 4.0], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test6"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    value = np.array([1, 2, 3], dtype=np.int32)
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "test7"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test8"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    value = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dtype = tf.float64
    dtype_hint = tf.float64
    name = "test9"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "test10"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
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
