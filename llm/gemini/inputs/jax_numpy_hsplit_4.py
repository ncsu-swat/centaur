
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array, split into 2 equal parts
    ary = np.arange(6, dtype=np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 2: 1D array, split at indices 2 and 4
    ary = np.arange(10, dtype=np.int32)
    indices_or_sections = np.array([2, 4], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 3: 2D array, split into 3 equal parts horizontally
    ary = np.random.randn(4, 9).astype(np.float32)
    indices_or_sections = np.array(3, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 4: 2D array, split at specific indices horizontally
    ary = np.random.randn(3, 8).astype(np.float64)
    indices_or_sections = np.array([2, 5, 7], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 5: 3D array, split into 2 equal parts horizontally
    ary = np.random.randn(2, 4, 3).astype(np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 6: 3D array, split at indices horizontally
    ary = np.random.randint(-50, 50, size=(2, 6, 2)).astype(np.int32)
    indices_or_sections = np.array([1, 4], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 7: 1D array with negative values, float64, split into 4
    ary = np.linspace(-10.0, 10.0, 12).astype(np.float64)
    indices_or_sections = np.array(4, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 8: 4D array, split into 2 equal parts horizontally
    ary = np.random.randn(2, 2, 2, 2).astype(np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 9: 2D array with booleans, split at index 3
    ary = np.random.choice([True, False], size=(2, 6))
    indices_or_sections = np.array([3], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 10: 2D array with large integers, split into 5 parts
    ary = np.random.randint(-100, 100, size=(5, 10)).astype(np.int64)
    indices_or_sections = np.array(5, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    return list_of_inputs

generated_inputs["jax.numpy.hsplit_4"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hsplit_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hsplit_4'.")


check_valid('jax.numpy.hsplit', generated_inputs['jax.numpy.hsplit_4'], lib="jax", suffix=4)
