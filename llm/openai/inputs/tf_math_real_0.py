
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_math_real_inputs():
    list_of_inputs = []
    
    # Input 1: Real tensor with negative values
    input_tensor = np.array([-2.25, -3.25, -4.75], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_negative"}
    list_of_inputs.append(input_dict)
    
    # Input 2: Real tensor with positive values
    input_tensor = np.array([2.25, 3.25, 4.75], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_positive"}
    list_of_inputs.append(input_dict)
    
    # Input 3: Complex tensor with real part
    input_tensor = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "complex_real"}
    list_of_inputs.append(input_dict)
    
    # Input 4: Complex tensor with imaginary part
    input_tensor = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "complex_imaginary"}
    list_of_inputs.append(input_dict)
    
    # Input 5: Complex tensor with mixed real and imaginary parts
    input_tensor = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "complex_mixed"}
    list_of_inputs.append(input_dict)
    
    # Input 6: Real tensor with zero values
    input_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_zero"}
    list_of_inputs.append(input_dict)
    
    # Input 7: Real tensor with decimal values
    input_tensor = np.array([1.5, 2.7, 3.9], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_decimal"}
    list_of_inputs.append(input_dict)
    
    # Input 8: Real tensor with large values
    input_tensor = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_large"}
    list_of_inputs.append(input_dict)
    
    # Input 9: Real tensor with small values
    input_tensor = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_small"}
    list_of_inputs.append(input_dict)
    
    # Input 10: Real tensor with negative values
    input_tensor = np.array([-1.5, -2.7, -3.9], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_negative_decimal"}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.math.real"] = tf_math_real_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.real'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.real', generated_inputs['tf.math.real'], lib="tf", suffix=0)
