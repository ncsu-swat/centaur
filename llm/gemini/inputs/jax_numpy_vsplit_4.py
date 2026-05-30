
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array split into 2 sections
    ary = np.random.randn(6).astype(np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array split into 2 sections
    ary = np.random.randn(4, 4).astype(np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array split at specific indices
    ary = np.random.randn(6, 3).astype(np.float32)
    indices_or_sections = np.array([2, 4], dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array split into 4 sections
    ary = np.random.randn(8, 2, 2).astype(np.float32)
    indices_or_sections = np.array(4, dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array split at specific indices
    ary = np.random.randn(10).astype(np.float32)
    indices_or_sections = np.array([3, 7], dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array split into 3 sections (integer type array)
    ary = np.random.randint(-10, 10, size=(12, 5)).astype(np.int32)
    indices_or_sections = np.array(3, dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array split at multiple indices (double precision)
    ary = np.random.randn(10, 10).astype(np.float64)
    indices_or_sections = np.array([1, 5, 9], dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array split into 2 sections
    ary = np.random.randn(4, 2, 2, 2).astype(np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array split into 3 sections
    ary = np.random.randn(9, 3, 3).astype(np.float32)
    indices_or_sections = np.array(3, dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array split at 1 index
    ary = np.random.randn(5, 5).astype(np.float32)
    indices_or_sections = np.array([2], dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vsplit_4"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vsplit_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vsplit_4'.")


check_valid('jax.numpy.vsplit', generated_inputs['jax.numpy.vsplit_4'], lib="jax", suffix=4)
