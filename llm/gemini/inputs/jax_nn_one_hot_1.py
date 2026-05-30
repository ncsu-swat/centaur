
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_one_hot_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, float32, axis -1
    list_of_inputs.append({
        "x": np.array([0, 1, 2], dtype=np.int32),
        "num_classes": 3,
        "dtype": np.dtype('float32'),
        "axis": -1
    })

    # Input 2: 1D array with negative and out-of-bounds indices, float64, axis -1
    list_of_inputs.append({
        "x": np.array([-1, 0, 3, 5], dtype=np.int32),
        "num_classes": 4,
        "dtype": np.dtype('float64'),
        "axis": -1
    })

    # Input 3: 2D array, float32, axis -1
    list_of_inputs.append({
        "x": np.array([[0, 1], [2, 3]], dtype=np.int32),
        "num_classes": 5,
        "dtype": np.dtype('float32'),
        "axis": -1
    })

    # Input 4: 2D array, float16, axis 0
    list_of_inputs.append({
        "x": np.array([[1, 2], [0, 1]], dtype=np.int32),
        "num_classes": 3,
        "dtype": np.dtype('float16'),
        "axis": 0
    })

    # Input 5: Scalar (0D array), float32, axis -1
    list_of_inputs.append({
        "x": np.array(2, dtype=np.int32),
        "num_classes": 5,
        "dtype": np.dtype('float32'),
        "axis": -1
    })

    # Input 6: 3D array, float32, axis 1
    list_of_inputs.append({
        "x": np.array([[[0, 1], [2, 0]], [[1, 1], [0, 2]]], dtype=np.int32),
        "num_classes": 3,
        "dtype": np.dtype('float32'),
        "axis": 1
    })

    # Input 7: 1D array, float64, axis 0
    list_of_inputs.append({
        "x": np.array([1, 0, 2], dtype=np.int64),
        "num_classes": 4,
        "dtype": np.dtype('float64'),
        "axis": 0
    })

    # Input 8: 2D array, float32, negative axis
    list_of_inputs.append({
        "x": np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int32),
        "num_classes": 3,
        "dtype": np.dtype('float32'),
        "axis": -2
    })

    # Input 9: Large num_classes, 1D array, float32, axis -1
    list_of_inputs.append({
        "x": np.array([5, 12, 19], dtype=np.int32),
        "num_classes": 20,
        "dtype": np.dtype('float32'),
        "axis": -1
    })

    # Input 10: 4D array, float32, axis -1
    list_of_inputs.append({
        "x": np.random.randint(0, 5, size=(2, 2, 2, 2), dtype=np.int32),
        "num_classes": 5,
        "dtype": np.dtype('float32'),
        "axis": -1
    })

    return list_of_inputs

generated_inputs["jax.nn.one_hot_1"] = generate_one_hot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.one_hot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.one_hot_1'.")


check_valid('jax.nn.one_hot', generated_inputs['jax.nn.one_hot_1'], lib="jax", suffix=1)
