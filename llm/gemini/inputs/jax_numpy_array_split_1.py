
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def array_split_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, split into 3 sections (even split)
    ary = np.random.randn(9).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 3,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, split into 4 sections (uneven split)
    ary = np.arange(10, dtype=np.int32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 4,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, split along axis 0
    ary = np.random.randn(6, 4).astype(np.float64)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, split along axis 1
    ary = np.random.randn(4, 7).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 3,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, split along axis 2
    ary = np.random.randint(-10, 10, size=(2, 3, 5)).astype(np.int64)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, split along negative axis -1 (which is axis 2)
    ary = np.random.randn(2, 4, 6).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 3,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float64 array, split along axis 3
    ary = np.random.randn(2, 2, 3, 4).astype(np.float64)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D bool array, split into 2 sections
    ary = np.array([True, False, True, False, True]).astype(np.bool_)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, split along negative axis -2 (which is axis 0)
    ary = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, split along axis 1 with 4 sections (uneven split)
    ary = np.random.randn(3, 10, 2).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 4,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.array_split_1"] = array_split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_split_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_split_1'.")


check_valid('jax.numpy.array_split', generated_inputs['jax.numpy.array_split_1'], lib="jax", suffix=1)
