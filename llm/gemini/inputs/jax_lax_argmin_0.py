
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argmin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, axis 0, index_dtype int32
    operand = np.array([4.5, 1.2, 3.3, 0.5, 2.1], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "axis": 0,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, axis 0, index_dtype int32
    operand = np.random.randn(3, 5).astype(np.float32)
    input_dict = {
        "operand": operand,
        "axis": 0,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32, axis 1, index_dtype int32
    operand = np.random.randn(4, 6).astype(np.float32)
    input_dict = {
        "operand": operand,
        "axis": 1,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32, negative values, axis 2, index_dtype int32
    operand = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    input_dict = {
        "operand": operand,
        "axis": 2,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32, axis 1, index_dtype int32
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "operand": operand,
        "axis": 1,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int32, negative values, axis 0, index_dtype int32
    operand = np.array([-10, 20, -30, 40], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "axis": 0,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32, axis 1, index_dtype int32
    operand = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "operand": operand,
        "axis": 1,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32, axis 2, index_dtype int32
    operand = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {
        "operand": operand,
        "axis": 2,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D int32, negative values, axis 3, index_dtype int32
    operand = np.random.randint(-50, 50, size=(2, 2, 3, 4, 2)).astype(np.int32)
    input_dict = {
        "operand": operand,
        "axis": 3,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 with duplicate minimums, axis 1, index_dtype int32
    operand = np.array([[1.0, 1.0, 2.0], [3.0, 0.5, 0.5]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "axis": 1,
        "index_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.argmin"] = argmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.argmin'.")


check_valid('jax.lax.argmin', generated_inputs['jax.lax.argmin'], lib="jax", suffix=0)
