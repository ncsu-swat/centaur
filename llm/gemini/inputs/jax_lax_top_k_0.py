
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_top_k_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, k=3, axis=-1
    operand = np.array([9.0, 3.0, 6.0, 4.0, 10.0], dtype=np.float32)
    input_dict = {"operand": operand, "k": 3, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative values, k=2, axis=0
    operand = np.array([-10, -3, -5, 0, 2, -1], dtype=np.int32)
    input_dict = {"operand": operand, "k": 2, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, k=2, axis=-1
    operand = np.random.randn(3, 5).astype(np.float32)
    input_dict = {"operand": operand, "k": 2, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, k=1, axis=0
    operand = np.random.randn(4, 4).astype(np.float64)
    input_dict = {"operand": operand, "k": 1, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, k=4, axis=1
    operand = np.random.randn(2, 5, 3).astype(np.float32)
    input_dict = {"operand": operand, "k": 4, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array, k=3, axis=-1
    operand = np.random.randint(0, 100, size=(4, 6)).astype(np.int32)
    input_dict = {"operand": operand, "k": 3, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D int64 array, k=2, axis=2
    operand = np.random.randint(-50, 50, size=(2, 2, 3, 2)).astype(np.int64)
    input_dict = {"operand": operand, "k": 2, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array with duplicate values, k=5, axis=0
    operand = np.array([5.0, 5.0, 2.0, 8.0, 8.0, 1.0, 9.0, 9.0], dtype=np.float32)
    input_dict = {"operand": operand, "k": 5, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array with negative and positive values, k=5, axis=-1
    operand = np.random.uniform(-10.0, 10.0, size=(3, 3, 6)).astype(np.float32)
    input_dict = {"operand": operand, "k": 5, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array, k=2, axis=-2
    operand = np.random.randn(4, 3).astype(np.float32)
    input_dict = {"operand": operand, "k": 2, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.top_k"] = jax_lax_top_k_inputs()

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
