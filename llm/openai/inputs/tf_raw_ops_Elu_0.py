
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def elu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    feature = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "name": "el1",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    feature = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {
        "name": "el2",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    feature = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {
        "name": "el3",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    feature = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    input_dict = {
        "name": "el4",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    feature = np.array([[-1000.0], [-1.0]], dtype=np.float32)
    input_dict = {
        "name": "el5",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    feature = np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "name": "el6",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    feature = np.array([1.0], dtype=np.float32)
    input_dict = {
        "name": "el7",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    feature = np.array([-1.0], dtype=np.float32)
    input_dict = {
        "name": "el8",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    feature = np.array([0.0], dtype=np.float32)
    input_dict = {
        "name": "el9",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    feature = np.array([[-1000.0, -100.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {
        "name": "el10",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = elu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Elu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Elu', generated_inputs['tf.raw_ops.Elu'], lib="tf", suffix=0)
