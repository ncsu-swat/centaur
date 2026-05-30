
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class StringInt(str):
    def __index__(self):
        return int(str.__str__(self))

def jax_nn_one_hot_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, float32
    x = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {
        "x": x,
        "num_classes": 3,
        "dtype": np.dtype(np.float32),
        "axis": StringInt("-1")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with some negative/out-of-bounds indices, float64
    x = np.array([[0, -1], [2, 3]], dtype=np.int32)
    input_dict = {
        "x": x,
        "num_classes": 3,
        "dtype": np.dtype(np.float64),
        "axis": StringInt("1")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, int32 dtype output
    x = np.random.randint(0, 5, size=(2, 2, 2)).astype(np.int32)
    input_dict = {
        "x": x,
        "num_classes": 5,
        "dtype": np.dtype(np.int32),
        "axis": StringInt("2")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D array (scalar), float16
    x = np.array(1, dtype=np.int32)
    input_dict = {
        "x": x,
        "num_classes": 4,
        "dtype": np.dtype(np.float16),
        "axis": StringInt("0")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with negative values, uint8
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    input_dict = {
        "x": x,
        "num_classes": 3,
        "dtype": np.dtype(np.uint8),
        "axis": StringInt("0")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger 2D array, float32
    x = np.random.randint(-1, 10, size=(5, 5)).astype(np.int32)
    input_dict = {
        "x": x,
        "num_classes": 8,
        "dtype": np.dtype(np.float32),
        "axis": StringInt("-1")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, float64, empty-like values (all out of bounds)
    x = np.array([10, 20, 30], dtype=np.int32)
    input_dict = {
        "x": x,
        "num_classes": 5,
        "dtype": np.dtype(np.float64),
        "axis": StringInt("1")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, float16
    x = np.random.randint(0, 4, size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {
        "x": x,
        "num_classes": 4,
        "dtype": np.dtype(np.float16),
        "axis": StringInt("3")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, uint16
    x = np.array([[1, 2, 1], [0, 0, 2]], dtype=np.int32)
    input_dict = {
        "x": x,
        "num_classes": 3,
        "dtype": np.dtype(np.uint16),
        "axis": StringInt("-2")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with large values, float32
    x = np.array([0, 99, 100, 2], dtype=np.int32)
    input_dict = {
        "x": x,
        "num_classes": 10,
        "dtype": np.dtype(np.float32),
        "axis": StringInt("-2")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.one_hot_2"] = jax_nn_one_hot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.one_hot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.one_hot_2'.")


check_valid('jax.nn.one_hot', generated_inputs['jax.nn.one_hot_2'], lib="jax", suffix=2)
