
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def negative_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar (positive int32)
    list_of_inputs.append({"x": np.int32(42)})

    # Input 2: 0D scalar (negative int64)
    list_of_inputs.append({"x": np.int64(-100)})

    # Input 3: 1D array of mixed integers (int32)
    list_of_inputs.append({"x": np.array([1, -2, 3, -4], dtype=np.int32)})

    # Input 4: 2D array of integers (int64)
    list_of_inputs.append({"x": np.array([[1, -2], [3, -4]], dtype=np.int64)})

    # Input 5: 3D array of integers (int16)
    list_of_inputs.append({"x": np.array([[[1], [2]], [[-3], [-4]]], dtype=np.int16)})

    # Input 6: 1D array of boundary values (int8)
    list_of_inputs.append({"x": np.array([-128, 0, 127], dtype=np.int8)})

    # Input 7: 1D array of unsigned integers (uint32)
    list_of_inputs.append({"x": np.array([0, 5, 4294967295], dtype=np.uint32)})

    # Input 8: 4D array of zeros (int32)
    list_of_inputs.append({"x": np.zeros((2, 2, 2, 2), dtype=np.int32)})

    # Input 9: Large random integers (int64)
    list_of_inputs.append({"x": np.random.randint(-100000, 100000, size=(5, 5), dtype=np.int64)})

    # Input 10: Single int16 boundary scalar
    list_of_inputs.append({"x": np.int16(-32768)})

    return list_of_inputs

generated_inputs["jax.numpy.negative_2"] = negative_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.negative_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.negative_2'.")


check_valid('jax.numpy.negative', generated_inputs['jax.numpy.negative_2'], lib="jax", suffix=2)
