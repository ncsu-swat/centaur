
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_compat_forward_compatible_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_dict = {
        "year": 2023,
        "month": 10,
        "day": 15
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    input_dict = {
        "year": 2024,
        "month": 1,
        "day": 1
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    input_dict = {
        "year": 2023,
        "month": 12,
        "day": 31
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    input_dict = {
        "year": 2022,
        "month": 6,
        "day": 15
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    input_dict = {
        "year": 2023,
        "month": 7,
        "day": 4
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    input_dict = {
        "year": 2023,
        "month": 2,
        "day": 29
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    input_dict = {
        "year": 2023,
        "month": 5,
        "day": 1
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    input_dict = {
        "year": 2024,
        "month": 3,
        "day": 10
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid
    input_dict = {
        "year": 2023,
        "month": 11,
        "day": 20
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid
    input_dict = {
        "year": 2023,
        "month": 9,
        "day": 5
    }
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs["tf.compat.forward_compatible"] = tf_compat_forward_compatible_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.compat.forward_compatible' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.forward_compatible'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.compat.forward_compatible', generated_inputs['tf.compat.forward_compatible'], lib="tf", suffix=0)
