
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def array_split_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, split into 3 sections (0D tensor)
    ary = np.random.randn(10).astype(np.float32)
    indices_or_sections = np.array(3)
    axis = 0
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 2: 1D int32 array, split at indices [2, 5, 7]
    ary = np.arange(12).astype(np.int32)
    indices_or_sections = np.array([2, 5, 7])
    axis = 0
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 3: 2D float64 array, split into 4 sections along axis 1
    ary = np.random.randn(6, 11).astype(np.float64)
    indices_or_sections = np.array(4)
    axis = 1
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 4: 2D float32 array, split at indices [3, 8] along axis 0
    ary = np.random.randn(12, 5).astype(np.float32)
    indices_or_sections = np.array([3, 8])
    axis = 0
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 5: 3D float32 array, split into 2 sections along axis 2
    ary = np.random.randn(3, 4, 5).astype(np.float32)
    indices_or_sections = np.array(2)
    axis = 2
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 6: 3D int64 array, split at indices [1, 3] along negative axis -1
    ary = np.random.randint(0, 10, size=(2, 3, 6)).astype(np.int64)
    indices_or_sections = np.array([1, 3])
    axis = -1
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 7: 4D float32 array, split into 3 sections along axis 1
    ary = np.random.randn(2, 7, 3, 4).astype(np.float32)
    indices_or_sections = np.array(3)
    axis = 1
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 8: 1D float32 array, split at index [5]
    ary = np.random.randn(15).astype(np.float32)
    indices_or_sections = np.array([5])
    axis = 0
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 9: 2D float32 array, split into 5 sections along axis -2
    ary = np.random.randn(9, 9).astype(np.float32)
    indices_or_sections = np.array(5)
    axis = -2
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    # Input 10: 5D float32 array, split at indices [2, 4] along axis 3
    ary = np.random.randn(2, 2, 2, 6, 2).astype(np.float32)
    indices_or_sections = np.array([2, 4])
    axis = 3
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.array_split_4"] = array_split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_split_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_split_4'.")


check_valid('jax.numpy.array_split', generated_inputs['jax.numpy.array_split_4'], lib="jax", suffix=4)
