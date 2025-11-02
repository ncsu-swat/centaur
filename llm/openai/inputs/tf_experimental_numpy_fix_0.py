
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experiment_numpy_fix_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.2, -2.7, 3.9])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([[1.5, -2.3], [3.7, -4.1]])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([[[1.2, 2.8], [3.1, 4.9]], [[5.7, 6.3], [7.4, 8.6]]])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([0.0, -1.5, 2.3])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([-0.9, 0.8, -1.2])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([1.0, 2.0, 3.0])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([-1.5, -2.5, -3.5])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([0.0, 0.0, 0.0])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[1.5, -2.3], [3.7, -4.1]])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-1.9, 2.1], [-3.2, 4.3]])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.fix"] = tf_experiment_numpy_fix_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.fix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.fix'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.fix', generated_inputs['tf.experimental.numpy.fix'], lib="tf", suffix=0)
