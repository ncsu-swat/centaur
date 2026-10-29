
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_gt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([1.0, -2.0, 3.5, 0.0], dtype=np.float32)
    y = np.array([0.0, -1.0, 3.5, -0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 arrays with identical shape
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 1], [4, 3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D broadcasting with same rank
    x = np.random.randn(3, 4).astype(np.float64)
    y = np.random.randn(1, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar arrays
    x = np.array(5, dtype=np.int64)
    y = np.array(3, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar with 3D array (scalar broadcast)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D float16 arrays with broadcasting along multiple dimensions
    x = np.random.randn(4, 1, 5).astype(np.float16)
    y = np.random.randn(1, 3, 5).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Boolean arrays
    x = np.array([[True, False], [True, False]], dtype=np.bool_)
    y = np.array([[False, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D uint8 arrays with edge values
    x = np.array([0, 128, 255], dtype=np.uint8)
    y = np.array([0, 100, 254], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D int8 arrays
    x = np.random.randint(-128, 127, size=(2, 2, 3, 3), dtype=np.int8)
    y = np.random.randint(-128, 127, size=(2, 2, 3, 3), dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D arrays with infinities and NaNs (float32)
    x = np.array([[np.inf, -np.inf], [np.nan, 1.0]], dtype=np.float32)
    y = np.array([[1000.0, -1000.0], [0.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 0D scalar x compared to 2D y
    x = np.array(-5, dtype=np.int16)
    y = np.array([[-10, 0], [5, -5]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.gt"] = lax_gt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gt'.")


check_valid('jax.lax.gt', generated_inputs['jax.lax.gt'], lib="jax", suffix=0)
