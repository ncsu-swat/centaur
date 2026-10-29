
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default axis -1
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = -1
    where = np.array([True, True, True, True, True], dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 2: 2D float32 array, axis 0, with partial masking
    x = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    where = np.array(
        [
            [True, False, True, True],
            [True, True, False, True],
            [False, True, True, True],
        ],
        dtype=bool,
    )
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 3: 2D float64 array, axis 1
    x = np.random.randn(5, 5).astype(np.float64)
    axis = 1
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 4: 3D float32 array, axis -2
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    axis = -2
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 5: 3D float32 array, broadcastable where mask
    x = np.random.randn(4, 3, 2).astype(np.float32)
    axis = 1
    where = np.array([[[True], [True], [False]]], dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 6: 4D float32 array, axis 2
    x = np.random.randn(2, 2, 4, 3).astype(np.float32)
    axis = 2
    where = np.ones((2, 2, 4, 3), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 7: 1D array with negative values
    x = np.array([-100.0, -50.0, -20.0, -10.0], dtype=np.float32)
    axis = 0
    where = np.array([True, True, True, False], dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 8: 2D float16 array, axis -1
    x = np.random.randn(4, 6).astype(np.float16)
    axis = -1
    where = np.ones((4, 6), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 9: 4D float64 array, axis 3
    x = np.random.randn(1, 3, 2, 5).astype(np.float64)
    axis = 3
    where = np.random.choice([True, False], size=(1, 3, 2, 5), p=[0.8, 0.2])
    where[:, :, :, 0] = True
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 10: 2D array with large dynamic range
    x = np.array([[1000.0, -1000.0], [0.0, 500.0]], dtype=np.float32)
    axis = 1
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    # Input 11: 3D float32 array, axis 0
    x = np.random.randn(3, 2, 2).astype(np.float32)
    axis = 0
    where = np.ones((3, 2, 2), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": copy.deepcopy(where)})

    return list_of_inputs


generated_inputs["jax.nn.log_softmax_1"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.log_softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.log_softmax_1'.")


check_valid('jax.nn.log_softmax', generated_inputs['jax.nn.log_softmax_1'], lib="jax", suffix=1)
