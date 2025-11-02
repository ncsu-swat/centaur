
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signbit_inputs():
    list_of_inputs = []
    
    # Input 1, valid - tensor with negative values
    x = np.array([-1.0, 2.0, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - tensor with positive values
    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - tensor with mixed values (positive and negative)
    x = np.array([-1.0, 2.0, -3.0, 4.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - scalar tensor
    x = np.array(5.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - 2D tensor with negative values
    x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - 3D tensor with negative values
    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - tensor with zero values
    x = np.array([0.0, 1.0, -2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - tensor with very small values (close to zero)
    x = np.array([1e-10, -1e-10, 2e-10])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - tensor with large negative values
    x = np.array([-1000.0, 500.0, -2000.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - tensor with floating point values (positive and negative)
    x = np.array([1.5, -2.7, 3.9, -4.1])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.signbit"] = tf_signbit_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.signbit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.signbit'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.signbit', generated_inputs['tf.experimental.numpy.signbit'], lib="tf", suffix=0)
