
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_lax_sub_inputs():
    list_of_inputs = []

    # 1. 1D float32 arrays of same shape
    x = np.array([1.0, -2.5, 3.0], dtype=np.float32)
    y = np.array([0.5, 1.5, -3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 2. 2D int32 arrays with broadcasting (same rank)
    x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    y = np.array([[1], [2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 3. 0D scalar arrays float64
    x = np.array(5.5, dtype=np.float64)
    y = np.array(-2.3, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 4. 0D scalar broadcast with 3D array int64
    x = np.ones((2, 3, 4), dtype=np.int64) * 10
    y = np.array(4, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 5. 3D float32 arrays with matching shapes
    x = np.random.randn(2, 4, 3).astype(np.float32)
    y = np.random.randn(2, 4, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 6. 2D complex64 arrays
    x = np.array([[1 + 2j, -3 + 4j], [5 - 6j, 7 + 8j]], dtype=np.complex64)
    y = np.array([[2 - 1j, 1 + 1j], [-3 + 2j, 0 + 4j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 7. 4D float64 arrays with singleton dimension broadcasting (same rank)
    x = np.zeros((2, 1, 4, 5), dtype=np.float64)
    y = np.ones((2, 3, 4, 1), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 8. 2D float32 arrays with negative values
    x = np.array([[-10.0, 20.0], [-30.0, 40.0]], dtype=np.float32)
    y = np.array([[5.0, -15.0], [25.0, -35.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 9. 1D int64 arrays
    x = np.array([100, -200, 300, -400], dtype=np.int64)
    y = np.array([50, 100, -150, 200], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 10. 3D complex128 arrays
    x = (
        np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)
    ).astype(np.complex128)
    y = (
        np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)
    ).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 11. 2D float64 arrays with rank-compatible broadcast
    x = np.ones((4, 1), dtype=np.float64)
    y = np.zeros((4, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.sub_1"] = jax_lax_sub_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sub_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sub_1'.")


check_valid('jax.lax.sub', generated_inputs['jax.lax.sub_1'], lib="jax", suffix=1)
