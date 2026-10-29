
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_lax_max_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([-1.5, 0.0, 2.3, -4.8, 5.2], dtype=np.float32)
    y = np.array([1.2, -0.5, 2.1, -3.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float64 arrays with negative values
    x = np.random.randn(3, 4).astype(np.float64)
    y = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D int32 arrays with same-rank broadcasting
    x = np.random.randint(-100, 100, size=(2, 1, 4), dtype=np.int32)
    y = np.random.randint(-100, 100, size=(2, 3, 1), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar float32 arrays
    x = np.array(3.14, dtype=np.float32)
    y = np.array(-2.71, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 4D int16 arrays
    x = np.random.randint(-50, 50, size=(2, 2, 3, 3), dtype=np.int16)
    y = np.random.randint(-50, 50, size=(2, 2, 3, 3), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 1D uint8 arrays
    x = np.array([0, 50, 128, 255], dtype=np.uint8)
    y = np.array([10, 20, 200, 100], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D complex64 arrays
    x = np.array([[1.0 + 2.0j, -3.0 + 4.0j], [0.0 - 1.0j, 5.0 + 0.0j]], dtype=np.complex64)
    y = np.array([[1.0 + 1.0j, -2.0 + 0.0j], [0.0 + 1.0j, 5.0 - 1.0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D complex128 arrays
    x = np.array([-2.5 + 3.5j, 4.0 - 1.2j, 0.0 + 0.0j], dtype=np.complex128)
    y = np.array([-2.5 + 1.5j, 3.9 + 5.0j, -0.1 + 0.0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D float16 arrays
    x = np.array([[-1.0, 2.0], [3.5, -4.5]], dtype=np.float16)
    y = np.array([[0.0, -2.0], [3.0, -5.0]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 3D int64 arrays
    x = np.random.randint(-1000, 1000, size=(3, 2, 2), dtype=np.int64)
    y = np.random.randint(-1000, 1000, size=(3, 2, 2), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 2D int8 arrays with broadcasting
    x = np.array([[-10, 20, -30]], dtype=np.int8)
    y = np.array([[5], [-15], [25]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.max_1"] = jax_lax_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.max_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.max_1'.")


check_valid('jax.lax.max', generated_inputs['jax.lax.max_1'], lib="jax", suffix=1)
