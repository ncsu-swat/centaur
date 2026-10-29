
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_lax_le_inputs():
    list_of_inputs = []

    # 1. Scalar arrays, float32
    x = np.array(1.5, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 2. 1D arrays, matching shapes, int32 with negative values
    x = np.array([-5, 0, 10, -2], dtype=np.int32)
    y = np.array([-3, 0, 8, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 3. 2D arrays, broadcastable shapes (same number of dimensions), float64
    x = np.random.randn(3, 1).astype(np.float64)
    y = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 4. 2D arrays, matching shapes, uint8
    x = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    y = np.array([[15, 20], [25, 45]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 5. 3D arrays, broadcastable shapes, float32
    x = np.random.randn(2, 1, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 6. 3D arrays, identical shapes, int64
    x = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    y = np.full((2, 3, 4), 12, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 7. 4D arrays, matching shapes, float16
    x = np.random.randn(1, 2, 2, 3).astype(np.float16)
    y = np.random.randn(1, 2, 2, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 8. Boolean arrays, 1D
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 9. 2D broadcastable with singleton dimension on y, int16
    x = np.array([[-10, 20, 30], [5, -15, 0]], dtype=np.int16)
    y = np.array([[0], [10]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 10. 5D arrays, matching shapes, float32
    x = np.zeros((1, 2, 1, 3, 2), dtype=np.float32)
    y = np.ones((1, 2, 1, 3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # 11. 1D empty arrays, int32
    x = np.array([], dtype=np.int32)
    y = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.le_1"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_1'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_1'], lib="jax", suffix=1)
