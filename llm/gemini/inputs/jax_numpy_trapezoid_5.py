
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D array with matching 1D x, axis=-1
    y = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, 1D x along axis=0
    y = np.random.randn(5, 8).astype(np.float32)
    x = np.linspace(0.0, 10.0, 5, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 2, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, same shape x along axis=1
    y = np.random.randn(4, 6).astype(np.float64)
    x = np.broadcast_to(
        np.linspace(1.0, 5.0, 6, dtype=np.float64), (4, 6)
    ).copy()
    input_dict = {"y": y, "x": x, "dx": 1, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, 1D x along axis=-1
    y = np.random.randn(3, 4, 5).astype(np.float32)
    x = np.array([0.0, 0.5, 1.5, 2.0, 3.5], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 3, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, matching shape x along axis=1
    y = np.random.randn(2, 5, 3).astype(np.float32)
    x = np.random.randn(2, 5, 3).astype(np.float32)
    input_dict = {"y": y, "x": x, "dx": -1, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with negative values and irregular spacing
    y = np.array([-2.5, -1.0, 0.0, 1.5, -0.5], dtype=np.float64)
    x = np.array([-5.0, -3.0, -1.0, 2.0, 4.0], dtype=np.float64)
    input_dict = {"y": y, "x": x, "dx": 1, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, 1D x along negative axis -2
    y = np.random.randn(6, 3).astype(np.float32)
    x = np.linspace(-2.0, 2.0, 6, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 5, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, 1D x along axis=2
    y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x = np.array([10.0, 20.0, 30.0, 50.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 2, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D integer arrays
    y = np.array([0, 3, 6, 2, 8, 1], dtype=np.int32)
    x = np.array([0, 1, 3, 6, 7, 10], dtype=np.int32)
    input_dict = {"y": y, "x": x, "dx": 1, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array with axis=0
    y = np.ones((8, 2, 2), dtype=np.float32)
    x = np.arange(8, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 4, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D complex array, 1D real x
    y = (np.random.randn(3, 7) + 1j * np.random.randn(3, 7)).astype(
        np.complex64
    )
    x = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.trapezoid_5"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_5'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_5'], lib="jax", suffix=5)
