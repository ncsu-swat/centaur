
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis 0, full boolean mask
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = 0
    where = np.array([True, True, True, False], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 2: 2D array, axis -1, float32, random values
    x = np.random.randn(3, 5).astype(np.float32)
    axis = -1
    where = np.ones((3, 5), dtype=np.bool_)
    where[0, 1] = False
    where[1, 3] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 3: 2D array, axis 0, float64
    x = np.random.randn(4, 6).astype(np.float64)
    axis = 0
    where = np.ones((4, 6), dtype=np.bool_)
    where[2, :] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 4: 3D array, axis 1, mixed negative and positive values
    x = np.random.uniform(-5.0, 5.0, size=(2, 4, 3)).astype(np.float32)
    axis = 1
    where = np.random.choice([True, False], size=(2, 4, 3), p=[0.8, 0.2])
    where[:, 0, :] = True  # Ensure at least one element is True along axis 1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 5: 3D array, axis -1, broadcastable where mask
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = -1
    where = np.array([[[True, True, True, False]]], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 6: 1D array with negative values only, axis -1
    x = np.array([-10.0, -20.0, -30.0, -5.0], dtype=np.float32)
    axis = -1
    where = np.array([True, False, True, True], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 7: 4D array, axis 2
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = 2
    where = np.ones((2, 3, 4, 5), dtype=np.bool_)
    where[:, :, 1, :] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 8: 2D array, axis 1, integer-valued floats
    x = np.array([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]], dtype=np.float32)
    axis = 1
    where = np.array([[True, True, False], [False, True, True]], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 9: 3D array, axis -2, float64
    x = np.random.randn(3, 5, 2).astype(np.float64)
    axis = -2
    where = np.ones((3, 5, 2), dtype=np.bool_)
    where[:, 0, :] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 10: 1D array, axis 0, all True where mask
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    axis = 0
    where = np.array([True, True, True], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    return list_of_inputs


generated_inputs["jax.nn.softmax_1"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.softmax_1'.")


check_valid('jax.nn.softmax', generated_inputs['jax.nn.softmax_1'], lib="jax", suffix=1)
