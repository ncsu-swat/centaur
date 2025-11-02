
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def conj_inputs():
    list_of_inputs = []
    
    # Input 1: Real tensor
    x = np.array([1, 2, 3, 4])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Complex tensor with real and imaginary parts
    x = np.array([1+2j, 3+4j, 5+6j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Complex tensor with negative values
    x = np.array([-1-2j, -3-4j, -5-6j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Scalar complex number
    x = np.array(1+2j)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with complex numbers
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor with complex numbers
    x = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Real tensor with floating point values
    x = np.array([1.5, 2.7, 3.9])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex tensor with zero imaginary part
    x = np.array([1+0j, 2+0j, 3+0j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Complex tensor with zero real part
    x = np.array([0+1j, 0+2j, 0+3j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Mixed complex and real tensor
    x = np.array([1+0j, 2.5+3j, 4+5j, 6.7+8j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.conj"] = conj_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.conj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.conj'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.conj', generated_inputs['tf.experimental.numpy.conj'], lib="tf", suffix=0)
