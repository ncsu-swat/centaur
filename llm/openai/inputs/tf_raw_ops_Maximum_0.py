
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_maximum_inputs():
    list_of_inputs = []
    
    # Input 1: Basic float32 tensors
    x = np.array([0., 0., 0., 0.], dtype=np.float32)
    y = np.array([-2., 0., 2., 5.], dtype=np.float32)
    input_dict = {
        "name": "maximum_1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Float64 tensors
    x = np.array([1.5, 2.7, 3.9], dtype=np.float64)
    y = np.array([1.2, 2.8, 3.1], dtype=np.float64)
    input_dict = {
        "name": "maximum_2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Int32 tensors with negative values
    x = np.array([-1, -5, 3], dtype=np.int32)
    y = np.array([0, -2, 4], dtype=np.int32)
    input_dict = {
        "name": "maximum_3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Int64 tensors with broadcasting
    x = np.array([-5., 0., 0., 0.], dtype=np.float64)
    y = np.array([-3.], dtype=np.float64)
    input_dict = {
        "name": "maximum_4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Uint8 tensors
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([0, 4, 5], dtype=np.uint8)
    input_dict = {
        "name": "maximum_5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Float32 with negative values and different dimensions
    x = np.array([[-1., 2., -3.], [4., -5., 6.]], dtype=np.float32)
    y = np.array([0., -1., 2.], dtype=np.float32)
    input_dict = {
        "name": "maximum_6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Int16 tensors with mixed positive and negative values
    x = np.array([1, -2, 3], dtype=np.int16)
    y = np.array([-1, 2, -3], dtype=np.int16)
    input_dict = {
        "name": "maximum_7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Half tensors
    x = np.array([1.5, 2.7], dtype=np.float16)
    y = np.array([1.2, 2.8], dtype=np.float16)
    input_dict = {
        "name": "maximum_8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Bfloat16 tensors
    x = np.array([1.5, 2.7], dtype=np.float32)
    y = np.array([1.2, 2.8], dtype=np.float32)
    input_dict = {
        "name": "maximum_9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex broadcasting with different shapes
    x = np.array([[-5., 0., 0., 0.], [-3., 1., 2., 3.]], dtype=np.float64)
    y = np.array([-3.], dtype=np.float64)
    input_dict = {
        "name": "maximum_10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Maximum"] = tf_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Maximum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Maximum', generated_inputs['tf.raw_ops.Maximum'], lib="tf", suffix=0)
