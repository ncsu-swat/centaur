
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def array_split_inputs():
    list_of_inputs = []

    # Input 1: 1D array split at (3, 5) along axis 0
    ary = np.arange(10, dtype=np.float32)
    indices_or_sections = (3, 5)
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array split at (2,) along axis 1
    ary = np.random.randn(4, 6).astype(np.float32)
    indices_or_sections = (2,)
    axis = 1
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array split at (2, 4) along axis 2
    ary = np.random.randn(2, 2, 8).astype(np.float32)
    indices_or_sections = (2, 4)
    axis = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative axis, 2D array split at (1, 3) along axis -1
    ary = np.random.randn(3, 5).astype(np.float32)
    indices_or_sections = (1, 3)
    axis = -1
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array split at (1, 2) along axis 0
    ary = np.random.randn(4, 2, 2, 2).astype(np.float32)
    indices_or_sections = (1, 2)
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tuple indices
    ary = np.random.randn(5).astype(np.float32)
    indices_or_sections = ()
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Indices within array size bounds
    ary = np.random.randn(3).astype(np.float32)
    indices_or_sections = (1, 2)
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean dtype array split at (2, 4) along axis 0
    ary = np.ones((5, 5), dtype=np.bool_)
    indices_or_sections = (2, 4)
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array split along negative axis -2
    ary = np.random.randn(6, 3).astype(np.float32)
    indices_or_sections = (1, 4)
    axis = -2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 2D array split at (2, 5, 8) along axis 0
    ary = np.random.randn(10, 2).astype(np.float64)
    indices_or_sections = (2, 5, 8)
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.array_split_3"] = array_split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_split_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_split_3'.")


check_valid('jax.numpy.array_split', generated_inputs['jax.numpy.array_split_3'], lib="jax", suffix=3)
