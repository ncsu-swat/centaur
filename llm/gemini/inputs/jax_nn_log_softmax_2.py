
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis along dimension 0
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = (0,)
    where = np.array([True, True, True, False], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 2: 2D float32 array, axis along last dimension
    x = np.random.randn(3, 5).astype(np.float32)
    axis = (-1,)
    where = np.ones((3, 5), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 3: 2D float64 array, axis along first dimension
    x = np.random.randn(4, 6).astype(np.float64)
    axis = (0,)
    where = np.random.rand(4, 6) > 0.2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 4: 3D float32 array, axis along single middle dimension
    x = np.random.randn(2, 4, 3).astype(np.float32)
    axis = (1,)
    where = np.ones((2, 4, 3), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 5: 3D float32 array, axis across multiple dimensions
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (1, 2)
    where = np.random.rand(2, 3, 4) > 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 6: 4D float32 array, axis along last two dimensions
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = (-2, -1)
    where = np.ones((2, 2, 3, 3), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 7: 2D float32 array with negative and positive values
    x = np.array([[-10.0, -5.0, 0.0], [5.0, 10.0, 15.0]], dtype=np.float32)
    axis = (1,)
    where = np.array(
        [[True, True, False], [True, False, True]], dtype=np.bool_
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 8: 3D float16 array, axis across non-contiguous dimensions
    x = np.random.randn(3, 2, 4).astype(np.float16)
    axis = (0, 2)
    where = np.ones((3, 2, 4), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 9: 1D float64 array with all negative values
    x = np.array([-100.0, -50.0, -20.0, -1.0], dtype=np.float64)
    axis = (-1,)
    where = np.array([True, True, True, True], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 10: 4D float32 array, axis along dimension 0
    x = np.random.randn(4, 2, 2, 2).astype(np.float32)
    axis = (0,)
    where = np.ones((4, 2, 2, 2), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 11: 2D float32 array, reducing all axes
    x = np.random.randn(3, 3).astype(np.float32)
    axis = (0, 1)
    where = np.array(
        [[True, False, True], [False, True, True], [True, True, False]],
        dtype=np.bool_,
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    return list_of_inputs


generated_inputs["jax.nn.log_softmax_2"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.log_softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.log_softmax_2'.")


check_valid('jax.nn.log_softmax', generated_inputs['jax.nn.log_softmax_2'], lib="jax", suffix=2)
