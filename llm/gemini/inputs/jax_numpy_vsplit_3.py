
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array, split at index 3
    ary = np.random.randn(6).astype(np.float32)
    indices_or_sections = (3,)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, split at index 4
    ary = np.random.randn(8, 4).astype(np.float32)
    indices_or_sections = (4,)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, split at multiple indices (2, 5, 8)
    ary = np.random.randn(10, 5).astype(np.float32)
    indices_or_sections = (2, 5, 8)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, split at index 3
    ary = np.random.randn(6, 3, 3).astype(np.float32)
    indices_or_sections = (3,)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, split at multiple indices (4, 8)
    ary = np.random.randn(12, 2, 2).astype(np.float32)
    indices_or_sections = (4, 8)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, split at indices (1, 3)
    ary = np.random.randn(5, 2, 2, 2).astype(np.float32)
    indices_or_sections = (1, 3)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int32 array, split at indices (3, 6)
    ary = np.random.randint(-10, 10, size=(9, 3)).astype(np.int32)
    indices_or_sections = (3, 6)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float64 array, split at indices (3, 6, 9)
    ary = np.random.randn(12).astype(np.float64)
    indices_or_sections = (3, 6, 9)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, split at indices (2, 4, 6)
    ary = np.random.randn(8, 1, 1, 1, 1).astype(np.float32)
    indices_or_sections = (2, 4, 6)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D bool array, split at index 1
    ary = (np.random.randn(4, 2) > 0).astype(np.bool_)
    indices_or_sections = (1,)
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vsplit_3"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vsplit_3'.")


check_valid('jax.numpy.vsplit', generated_inputs['jax.numpy.vsplit_3'], lib="jax", suffix=3)
