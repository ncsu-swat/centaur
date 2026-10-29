
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, default axis -1
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1.0, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with irregular spacing, float64, axis 0
    y = np.sin(np.linspace(0.0, np.pi, 20, dtype=np.float64))
    x = np.sort(np.random.uniform(0.0, np.pi, size=20)).astype(np.float64)
    input_dict = {"y": y, "x": x, "dx": 0.5, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, integrating along axis 0 with 1D x
    y = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 2.0, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, integrating along axis 1 with 1D x
    y = np.random.randn(5, 8).astype(np.float64)
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    input_dict = {"y": y, "x": x, "dx": 0.2857, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with 2D x of matching shape, axis -1
    y = np.ones((4, 6), dtype=np.float32)
    x = np.tile(np.linspace(0.0, 10.0, 6, dtype=np.float32), (4, 1))
    input_dict = {"y": y, "x": x, "dx": 1.5, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, integrating along axis -2 with 1D x
    y = np.random.uniform(-5.0, 5.0, size=(2, 6, 4)).astype(np.float32)
    x = np.array([0.0, 0.5, 1.5, 3.0, 4.0, 6.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 0.5, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with matching 3D x, axis 1
    y = np.random.randn(3, 5, 2).astype(np.float64)
    x = np.broadcast_to(
        np.linspace(0.0, 1.0, 5, dtype=np.float64)[:, None], (3, 5, 2)
    ).copy()
    input_dict = {"y": y, "x": x, "dx": 0.25, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with negative values and negative coordinates
    y = np.array([-3.0, -1.0, 0.0, 2.0, -1.0], dtype=np.float32)
    x = np.array([-10.0, -5.0, -2.0, -1.0, 0.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 2.5, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, integrating along axis 2
    y = np.random.randn(2, 3, 7, 2).astype(np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 7, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1.0, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with only two elements (minimal interval)
    y = np.array([3.5, 7.2], dtype=np.float64)
    x = np.array([1.0, 4.0], dtype=np.float64)
    input_dict = {"y": y, "x": x, "dx": 3.0, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array, negative axis -2 with 1D x
    y = np.random.randn(4, 3).astype(np.float64)
    x = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {"y": y, "x": x, "dx": 1.0, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.integrate.trapezoid_1"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.integrate.trapezoid_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.integrate.trapezoid_1'.")


check_valid('jax.scipy.integrate.trapezoid', generated_inputs['jax.scipy.integrate.trapezoid_1'], lib="jax", suffix=1)
