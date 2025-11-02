
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_isposinf_inputs():
    list_of_inputs = []
    
    # Input 1: 1D array with positive infinity
    x = np.array([1.0, np.inf, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D array with positive infinity
    x = np.array([[1.0, np.inf], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D array with negative infinity
    x = np.array([-np.inf, 1.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D array with mixed infinities
    x = np.array([[np.inf, -np.inf], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar with positive infinity
    x = np.array(1.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: scalar with negative infinity
    x = np.array(-1.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D array with float values including infinities
    x = np.array([np.inf, np.nan, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D array with positive infinity
    x = np.array([[[np.inf, 1.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 1D array with all positive infinity values
    x = np.array([np.inf, np.inf, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D array with all negative infinity values
    x = np.array([[-np.inf, -np.inf], [-np.inf, -np.inf]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isposinf"] = generate_isposinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.isposinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isposinf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.isposinf', generated_inputs['tf.experimental.numpy.isposinf'], lib="tf", suffix=0)
