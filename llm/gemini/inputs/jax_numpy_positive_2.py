
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def positive_inputs():
    list_of_inputs = []

    # Input 1: 0D integer scalar (np.int32)
    x = np.int32(-10)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array of np.int32 with positive, negative, and zero values
    x = np.array([-5, 0, 5], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of np.int64
    x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array of np.int16
    x = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array of np.int8 with minimum and maximum values
    x = np.array([-128, 0, 127], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D array of np.uint32 (unsigned integers)
    x = np.array([[10, 20], [30, 40]], dtype=np.uint32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D array of np.int32 with random integers
    x = np.random.randint(-100, 100, size=(2, 2, 3, 3)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Scalar np.int64
    x = np.int64(99999)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array of np.uint8
    x = np.array([0, 128, 255], dtype=np.uint8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array of np.int64 with large values
    x = np.array([[-9223372036854775807, 9223372036854775807]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.positive_2"] = positive_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.positive_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.positive_2'.")


check_valid('jax.numpy.positive', generated_inputs['jax.numpy.positive_2'], lib="jax", suffix=2)
