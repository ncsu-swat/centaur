
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vsplit_inputs():
    list_of_inputs = []

    # Input 1: 2D array split at index 3
    ary = np.random.randn(6, 2).astype(np.float32)
    indices_or_sections = [3]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array split at indices 1 and 3
    ary = np.random.randn(4, 4).astype(np.float32)
    indices_or_sections = [1, 3]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array float64 split at index 2
    ary = np.random.randn(5, 2).astype(np.float64)
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array split at indices 2 and 4
    ary = np.random.randn(6, 2, 2).astype(np.float32)
    indices_or_sections = [2, 4]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array split at multiple indices
    ary = np.arange(30).reshape(10, 3).astype(np.int32)
    indices_or_sections = [2, 5, 8]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array split at indices 1, 2, 3
    ary = np.random.randn(4, 2, 2, 2).astype(np.float32)
    indices_or_sections = [1, 2, 3]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with negative values, split at index 2
    ary = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D boolean array, split at index 1
    ary = (np.random.randn(2, 2) > 0)
    indices_or_sections = [1]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D complex array, split at index 3
    ary = (np.random.randn(5, 3, 3) + 1j * np.random.randn(5, 3, 3)).astype(np.complex64)
    indices_or_sections = [3]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 2D array, split at 10, 20, 30
    ary = np.random.randn(40, 10).astype(np.float32)
    indices_or_sections = [10, 20, 30]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vsplit_2"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vsplit_2'.")


check_valid('jax.numpy.vsplit', generated_inputs['jax.numpy.vsplit_2'], lib="jax", suffix=2)
