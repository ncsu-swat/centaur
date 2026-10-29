
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def top_k_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default-like axis -1
    operand = np.array([9.0, 3.0, 6.0, 4.0, 10.0], dtype=np.float32)
    input_dict = {"operand": operand, "k": 3, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative values, axis 0
    operand = np.array([-10, -3, 0, -25, 12, -4, 8], dtype=np.int32)
    input_dict = {"operand": operand, "k": 1, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis 1
    operand = np.random.randn(4, 6).astype(np.float32)
    input_dict = {"operand": operand, "k": 2, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, axis 0
    operand = np.random.randn(8, 4).astype(np.float64)
    input_dict = {"operand": operand, "k": 4, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int64 array with duplicates, k equals dimension length along axis -1
    operand = np.array(
        [[1, 3, 3, 2, 5], [10, 20, 20, 5, 0]], dtype=np.int64
    )
    input_dict = {"operand": operand, "k": 5, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis 2
    operand = np.random.randn(2, 3, 5).astype(np.float32)
    input_dict = {"operand": operand, "k": 2, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, axis 0
    operand = np.random.randn(5, 3, 2).astype(np.float32)
    input_dict = {"operand": operand, "k": 3, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array, axis 1
    operand = np.random.randint(-100, 100, size=(2, 6, 4), dtype=np.int32)
    input_dict = {"operand": operand, "k": 1, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, axis -1
    operand = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {"operand": operand, "k": 3, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float64 array, axis -2
    operand = np.random.randn(2, 4, 5, 3).astype(np.float64)
    input_dict = {"operand": operand, "k": 2, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int64 array, k=4
    operand = np.array([100, 250, 10, 500, 0, 75], dtype=np.int64)
    input_dict = {"operand": operand, "k": 4, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.top_k"] = top_k_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.top_k' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.top_k'.")


check_valid('jax.lax.top_k', generated_inputs['jax.lax.top_k'], lib="jax", suffix=0)
