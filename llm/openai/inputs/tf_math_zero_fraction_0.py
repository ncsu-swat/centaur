
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_math_zero_fraction_inputs():
    list_of_inputs = []
    
    # Input 1: Empty tensor
    value = np.array([], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_1"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: All zeros tensor
    value = np.array([0, 0, 0], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_2"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: Mixed values with some zeros
    value = np.array([1, 0, 3, 0, 5], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_3"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: Multi-dimensional tensor with zeros
    value = np.array([[1, 0], [0, 3]], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_4"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: All non-zero values
    value = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_5"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: Negative values with zeros
    value = np.array([-1, 0, -3, 0], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_6"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: Single element tensor
    value = np.array([0], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_7"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: Large tensor with many zeros
    value = np.random.choice([0, 1], size=(1000,), p=[0.5, 0.5])
    input_dict = {
        "value": value,
        "name": "test_8"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: Tensor with negative values
    value = np.array([-1, -2, -3], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_9"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: Float tensor with NaN values
    value = np.array([np.nan, np.nan, 0], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_10"
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.math.zero_fraction"] = tf_math_zero_fraction_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.zero_fraction' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.zero_fraction'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.zero_fraction', generated_inputs['tf.math.zero_fraction'], lib="tf", suffix=0)
