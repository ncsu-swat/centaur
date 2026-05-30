
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def split_inputs():
    list_of_inputs = []

    # Input 1: 1D array, split at indices (3, 6) along axis 0
    ary = np.arange(10, dtype=np.int32)
    indices_or_sections = (3, 6)
    axis = 0
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 2: 2D float32 array, split along axis 0
    ary = np.random.randn(6, 4).astype(np.float32)
    indices_or_sections = (2, 4)
    axis = 0
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 3: 2D float32 array, split along axis 1
    ary = np.random.randn(4, 5).astype(np.float32)
    indices_or_sections = (1, 3)
    axis = 1
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 4: 3D float32 array, split along axis 2
    ary = np.random.randn(2, 3, 4).astype(np.float32)
    indices_or_sections = (2,)
    axis = 2
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 5: 1D int64 array, split along axis 0
    ary = np.arange(5, dtype=np.int64)
    indices_or_sections = (1, 2, 3, 4)
    axis = 0
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 6: 2D float64 array, split along axis 0
    ary = np.random.randn(3, 3).astype(np.float64)
    indices_or_sections = (1,)
    axis = 0
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 7: 3D float32 array, split along axis 1
    ary = np.random.randn(2, 4, 2).astype(np.float32)
    indices_or_sections = (1, 2)
    axis = 1
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 8: 4D float32 array, split along axis 3
    ary = np.random.randn(2, 2, 2, 6).astype(np.float32)
    indices_or_sections = (2, 4)
    axis = 3
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 9: 2D float32 array, negative axis split
    ary = np.random.randn(3, 5).astype(np.float32)
    indices_or_sections = (2, 3)
    axis = -1
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    # Input 10: 1D boolean array, split along axis 0
    ary = np.array([True, False, True, False, True], dtype=bool)
    indices_or_sections = (1, 3)
    axis = 0
    list_of_inputs.append({
        "ary": copy.deepcopy(ary),
        "indices_or_sections": indices_or_sections,
        "axis": axis
    })

    return list_of_inputs

generated_inputs["jax.numpy.split_3"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.split_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.split_3'.")


check_valid('jax.numpy.split', generated_inputs['jax.numpy.split_3'], lib="jax", suffix=3)
