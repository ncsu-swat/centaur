
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sici_inputs():
    list_of_inputs = []

    # Input 1: 0-D tensor (scalar) with positive float32
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D tensor with positive float32
    x = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2-D tensor of shape (2, 3) with positive float32
    x = np.array([[1.0, 2.5, 3.2], [4.1, 0.8, 1.5]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3-D tensor of shape (2, 2, 2) with positive float64
    x = np.array([[[1.2, 2.3], [3.4, 4.5]], [[5.6, 6.7], [7.8, 8.9]]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1-D tensor with large positive values
    x = np.array([10.0, 50.0, 100.0, 500.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1-D tensor with small positive values close to zero
    x = np.array([1e-5, 1e-3, 0.01, 0.1], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4-D tensor with positive values in float32
    x = np.random.uniform(0.1, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2-D tensor with float64
    x = np.random.uniform(1.0, 20.0, size=(3, 4)).astype(np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1-D tensor with a single positive element
    x = np.array([3.14159], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5-D tensor with positive values in float32
    x = np.random.uniform(0.5, 5.0, size=(2, 1, 3, 1, 2)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.sici"] = sici_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.sici' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.sici'.")


check_valid('jax.scipy.special.sici', generated_inputs['jax.scipy.special.sici'], lib="jax", suffix=0)
