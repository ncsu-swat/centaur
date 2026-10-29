
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, axis=-1
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 2: 1D arrays, float64, axis=0, negative and positive values
    y = np.array([-2.5, 0.0, 3.5, -1.0, 4.0], dtype=np.float64)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    dx = np.array(1.0, dtype=np.float64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 3: 2D array, integration along axis=0
    y = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = np.array([0.0, 0.5, 1.5], dtype=np.float32)
    dx = np.array(0.5, dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 4: 2D array, integration along axis=1
    y = np.random.randn(4, 6).astype(np.float32)
    x = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    dx = np.array(0.2, dtype=np.float32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 5: 2D array with 2D x of matching shape, axis=-1
    y = np.ones((3, 5), dtype=np.float64)
    x = np.tile(np.linspace(0.0, 2.0, 5, dtype=np.float64), (3, 1))
    dx = np.array(0.5, dtype=np.float64)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 6: 3D array, axis=0
    y = np.random.uniform(-1.0, 1.0, size=(5, 3, 2)).astype(np.float32)
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 7: 3D array, axis=1
    y = np.random.randn(2, 8, 4).astype(np.float32)
    x = np.linspace(-3.0, 3.0, 8, dtype=np.float32)
    dx = np.array(0.75, dtype=np.float32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 8: 3D array, axis=2
    y = np.random.randn(3, 4, 7).astype(np.float64)
    x = np.array([0.1, 0.5, 0.9, 1.2, 2.0, 2.5, 3.0], dtype=np.float64)
    dx = np.array(0.1, dtype=np.float64)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 9: 4D array, axis=-2
    y = np.random.randn(2, 3, 5, 2).astype(np.float32)
    x = np.linspace(0.0, 10.0, 5, dtype=np.float32)
    dx = np.array(2.5, dtype=np.float32)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 10: 1D array with non-uniform irregular spacing, axis=0
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    x = np.array([0.0, 0.1, 0.4, 0.9, 1.6, 2.5], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.numpy.trapezoid_6"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_6'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_6'], lib="jax", suffix=6)
