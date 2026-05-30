
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array, even split
    ary = np.arange(6, dtype=np.float32)
    indices_or_sections = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, larger size
    ary = np.arange(12, dtype=np.int32)
    indices_or_sections = 3
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float64
    ary = np.random.randn(2, 4).astype(np.float64)
    indices_or_sections = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, splitting by 3
    ary = np.random.randn(3, 9).astype(np.float32)
    indices_or_sections = 3
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, splitting axis 1 (size 6) by 2
    ary = np.random.randn(2, 6, 4).astype(np.float32)
    indices_or_sections = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, splitting axis 1 (size 8) by 4
    ary = np.random.randn(2, 8, 2, 2).astype(np.float32)
    indices_or_sections = 4
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with negative float values
    ary = np.random.uniform(-10, 10, (4, 10)).astype(np.float32)
    indices_or_sections = 5
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with negative integers
    ary = np.array([-10, -5, 0, 5, 10, 15, 20, 25], dtype=np.int64)
    indices_or_sections = 4
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D boolean array
    ary = np.random.choice([True, False], size=(2, 6))
    indices_or_sections = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array
    ary = np.random.randn(1, 12, 2, 2, 2).astype(np.float32)
    indices_or_sections = 6
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.hsplit_1"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hsplit_1'.")


check_valid('jax.numpy.hsplit', generated_inputs['jax.numpy.hsplit_1'], lib="jax", suffix=1)
