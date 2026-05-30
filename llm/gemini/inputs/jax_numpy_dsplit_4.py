
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dsplit_inputs():
    list_of_inputs = []

    # Input 1: shape (2, 2, 4), split into 2 equal parts
    ary = np.random.randn(2, 2, 4).astype(np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 2: shape (1, 1, 6), split into 3 equal parts
    ary = np.random.randint(-10, 10, size=(1, 1, 6)).astype(np.int32)
    indices_or_sections = np.array(3, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 3: shape (3, 4, 8), split into 4 equal parts
    ary = np.random.randn(3, 4, 8).astype(np.float64)
    indices_or_sections = np.array(4, dtype=np.int64)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 4: shape (2, 3, 5), split at indices 1 and 3
    ary = np.random.randn(2, 3, 5).astype(np.float32)
    indices_or_sections = np.array([1, 3], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 5: shape (4, 2, 10), split at indices 2, 5, 8
    ary = np.random.randn(4, 2, 10).astype(np.float32)
    indices_or_sections = np.array([2, 5, 8], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 6: shape (1, 1, 4), split into 1 part
    ary = np.random.randint(0, 5, size=(1, 1, 4)).astype(np.int64)
    indices_or_sections = np.array(1, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 7: 4D array (2, 2, 2, 6), split axis 2 (size 2) into 2 parts
    ary = np.random.randn(2, 2, 2, 6).astype(np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 8: 4D array (1, 3, 12, 5), split axis 2 (size 12) at indices 3, 6, 9
    ary = np.random.randn(1, 3, 12, 5).astype(np.float32)
    indices_or_sections = np.array([3, 6, 9], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 9: shape (5, 5, 5), split axis 2 (size 5) at indices 1, 4
    ary = np.random.randint(-5, 5, size=(5, 5, 5)).astype(np.int32)
    indices_or_sections = np.array([1, 4], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 10: shape (2, 1, 15), split into 5 equal parts
    ary = np.random.randn(2, 1, 15).astype(np.float64)
    indices_or_sections = np.array(5, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    return list_of_inputs

generated_inputs["jax.numpy.dsplit_4"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dsplit_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dsplit_4'.")


check_valid('jax.numpy.dsplit', generated_inputs['jax.numpy.dsplit_4'], lib="jax", suffix=4)
