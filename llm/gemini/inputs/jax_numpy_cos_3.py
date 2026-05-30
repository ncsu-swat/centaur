
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_cos_inputs():
    list_of_inputs = []

    # Input 1: python integer scalar 0
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative python integer scalar
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: numpy int32 scalar
    input_dict = {"x": np.int32(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D numpy array of int32
    input_dict = {"x": np.array([0, 1, -1, 2], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D numpy array of int64
    input_dict = {"x": np.array([[1, -2], [3, -4]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 numpy array
    input_dict = {"x": np.array([0, 255, 128], dtype=np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D numpy array of int16
    input_dict = {"x": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large positive python integer scalar
    input_dict = {"x": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0D numpy array of int32
    input_dict = {"x": np.array(-5, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array of random integers
    input_dict = {"x": np.random.randint(-100, 100, size=(3, 3), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cos_3"] = jax_numpy_cos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cos_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cos_3'.")


check_valid('jax.numpy.cos', generated_inputs['jax.numpy.cos_3'], lib="jax", suffix=3)
