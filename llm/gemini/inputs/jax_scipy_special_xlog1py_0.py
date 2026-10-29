
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def xlog1py_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32, positive values
    x = np.array(2.5, dtype=np.float32)
    y = np.array(3.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 0D special case (x=0, y=-1)
    x = np.array(0.0, dtype=np.float64)
    y = np.array(-1.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D arrays, float32
    x = np.array([0.0, 1.5, -2.0, 3.2], dtype=np.float32)
    y = np.array([-1.0, 0.5, 2.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 1D array with y in (-1, 0)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([-0.5, -0.2, -0.9], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 2D arrays, same shape
    x = np.random.uniform(0.1, 5.0, size=(3, 4)).astype(np.float32)
    y = np.random.uniform(-0.5, 5.0, size=(3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 2D broadcasting (x is (3, 1), y is (1, 4))
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 3D arrays with float64
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.abs(np.random.randn(2, 3, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 3D with 1D broadcasting
    x = np.random.uniform(-2.0, 2.0, size=(2, 4, 3)).astype(np.float32)
    y = np.array([0.1, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D arrays, float32
    x = np.ones((2, 2, 3, 3), dtype=np.float32)
    y = np.full((2, 2, 3, 3), 4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty arrays
    x = np.empty((0, 5), dtype=np.float32)
    y = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Large positive values of y
    x = np.array([1e-5, 2.0, 10.0], dtype=np.float64)
    y = np.array([1e6, 1e8, 1e10], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 12: Integer tensor inputs
    x = np.array([[0, 2], [3, 5]], dtype=np.int32)
    y = np.array([[0, 1], [2, 3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.scipy.special.xlog1py"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.xlog1py' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.xlog1py'.")


check_valid('jax.scipy.special.xlog1py', generated_inputs['jax.scipy.special.xlog1py'], lib="jax", suffix=0)
