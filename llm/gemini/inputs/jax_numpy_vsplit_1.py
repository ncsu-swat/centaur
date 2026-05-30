
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vsplit_inputs():
    list_of_inputs = []

    # Input 1: 2D array, split into 2 parts
    ary = np.random.randn(6, 4).astype(np.float32)
    indices_or_sections = 2
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 2: 2D array, split into 3 parts
    ary = np.random.randn(6, 4).astype(np.float32)
    indices_or_sections = 3
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 3: 2D array, integer type, split into 4 parts
    ary = np.random.randint(-10, 10, size=(12, 5)).astype(np.int32)
    indices_or_sections = 4
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 4: 3D array, float64 type, split into 2 parts
    ary = np.random.randn(8, 4, 2).astype(np.float64)
    indices_or_sections = 2
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 5: 3D array, float64 type, split into 4 parts
    ary = np.random.randn(8, 4, 2).astype(np.float64)
    indices_or_sections = 4
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 6: 4D array, split into 5 parts
    ary = np.random.randn(10, 3, 3, 3).astype(np.float32)
    indices_or_sections = 5
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 7: 2D array, int64 type, split into 3 parts
    ary = np.random.randint(0, 100, size=(15, 2)).astype(np.int64)
    indices_or_sections = 3
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 8: 3D array, split into 3 parts
    ary = np.random.randn(9, 9, 9).astype(np.float32)
    indices_or_sections = 3
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 9: 2D array, split into 2 parts (each of size 1 along axis 0)
    ary = np.random.randn(2, 10).astype(np.float32)
    indices_or_sections = 2
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 10: 4D array, split into 6 parts
    ary = np.random.randn(12, 2, 2, 2).astype(np.float32)
    indices_or_sections = 6
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    return list_of_inputs

generated_inputs["jax.numpy.vsplit_1"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vsplit_1'.")


check_valid('jax.numpy.vsplit', generated_inputs['jax.numpy.vsplit_1'], lib="jax", suffix=1)
