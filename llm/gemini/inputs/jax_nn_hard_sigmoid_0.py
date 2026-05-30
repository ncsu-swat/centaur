
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hard_sigmoid_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 array with negative/positive values
    x = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array (matrix)
    x = np.random.uniform(-5.0, 5.0, size=(4, 4)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 3)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D (scalar) float32 array
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 array with wide range
    x = np.linspace(-6.0, 6.0, num=20, dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array (representing image-like data)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float16 array
    x = np.random.uniform(-3.0, 3.0, size=(5, 5)).astype(np.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array containing all zeros
    x = np.zeros((10,), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D float32 array
    x = np.random.randn(100, 100).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.hard_sigmoid"] = hard_sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.hard_sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.hard_sigmoid'.")


check_valid('jax.nn.hard_sigmoid', generated_inputs['jax.nn.hard_sigmoid'], lib="jax", suffix=0)
