
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, default axis -1
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 2: 1D arrays with irregular spacing
    y = np.array([0.5, -1.2, 3.4, 2.1, -0.8], dtype=np.float64)
    x = np.array([0.0, 0.2, 0.7, 1.5, 2.0], dtype=np.float64)
    dx = np.array(0.5, dtype=np.float64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 3: 2D arrays, integrate along axis 0 with 1D x
    y = np.ones((5, 4), dtype=np.float32)
    x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    dx = np.array(0.25, dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 4: 2D arrays, integrate along axis 1 with 1D x
    y = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 5: 2D arrays, both x and y having the same 2D shape
    y = np.random.randn(4, 6).astype(np.float32)
    x = np.broadcast_to(
        np.linspace(0.0, 5.0, 6, dtype=np.float32), (4, 6)
    ).copy()
    dx = np.array(1.0, dtype=np.float32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 6: 3D arrays, integrate along axis 1 with 1D x
    y = np.random.uniform(-5.0, 5.0, size=(2, 8, 3)).astype(np.float32)
    x = np.linspace(-2.0, 2.0, 8, dtype=np.float32)
    dx = np.array(0.5, dtype=np.float32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 7: 3D arrays, integrate along axis 0 with 3D x
    y = np.random.randn(3, 4, 5).astype(np.float64)
    x = np.random.randn(3, 4, 5).astype(np.float64)
    dx = np.array(0.1, dtype=np.float64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 8: 3D arrays, integrate along axis 2
    y = np.sin(np.linspace(0, np.pi, 24).reshape(2, 3, 4)).astype(np.float32)
    x = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    dx = np.array(0.25, dtype=np.float32)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 9: 4D arrays, integrate along axis -2
    y = np.ones((2, 3, 5, 2), dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    dx = np.array(2.0, dtype=np.float32)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 10: Negative axis with 2D array
    y = np.random.exponential(scale=2.0, size=(6, 7)).astype(np.float32)
    x = np.linspace(-3.0, 3.0, 6, dtype=np.float32)
    dx = np.array(1.2, dtype=np.float32)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    # Input 11: 1D arrays with float16
    y = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    x = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float16)
    dx = np.array(0.5, dtype=np.float16)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.numpy.trapezoid_3"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_3'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_3'], lib="jax", suffix=3)
