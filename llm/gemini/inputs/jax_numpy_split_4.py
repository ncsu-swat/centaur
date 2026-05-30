
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def split_inputs():
    list_of_inputs = []

    # Input 1: 1-D array split into 3 equal parts
    ary = np.arange(9, dtype=np.float32)
    indices_or_sections = np.array(3)
    axis = 0
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 2: 1-D array split at specific indices
    ary = np.arange(9, dtype=np.int32)
    indices_or_sections = np.array([2, 7])
    axis = 0
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 3: 2-D array split along axis 1 into 2 equal parts
    ary = np.random.randn(4, 6).astype(np.float32)
    indices_or_sections = np.array(2)
    axis = 1
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 4: 2-D array split along axis 0 into 4 equal parts
    ary = np.random.randn(8, 4).astype(np.float64)
    indices_or_sections = np.array(4)
    axis = 0
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 5: 2-D array split along negative axis (-1) at specific indices
    ary = np.random.randn(3, 9).astype(np.float32)
    indices_or_sections = np.array([3, 6])
    axis = -1
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 6: 3-D array split along axis 2 into 2 equal parts
    ary = np.random.randint(0, 10, size=(2, 2, 4)).astype(np.int64)
    indices_or_sections = np.array(2)
    axis = 2
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 7: 3-D array split along axis 0 at index 1
    ary = np.random.randn(3, 2, 2).astype(np.float32)
    indices_or_sections = np.array([1])
    axis = 0
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 8: 1-D array of float64 split into 5 equal parts
    ary = np.ones(10, dtype=np.float64)
    indices_or_sections = np.array(5)
    axis = 0
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 9: 4-D array split along axis 3 into 3 equal parts
    ary = np.random.randn(2, 2, 2, 6).astype(np.float32)
    indices_or_sections = np.array(3)
    axis = 3
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 10: 2-D array split along axis 1 at indices 1 and 3
    ary = np.random.randn(3, 5).astype(np.float32)
    indices_or_sections = np.array([1, 3])
    axis = 1
    list_of_inputs.append({
        "ary": ary,
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    return list_of_inputs

generated_inputs["jax.numpy.split_4"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.split_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.split_4'.")


check_valid('jax.numpy.split', generated_inputs['jax.numpy.split_4'], lib="jax", suffix=4)
