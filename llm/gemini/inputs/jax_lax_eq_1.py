
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_eq_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D float64 arrays
    x = np.array([[1.0, -2.0], [3.5, 4.0]], dtype=np.float64)
    y = np.array([[1.0, 2.0], [3.5, -4.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 1D int32 arrays with negative values
    x = np.array([-1, 0, 100], dtype=np.int32)
    y = np.array([-1, 1, 100], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 3D int64 arrays
    x = np.ones((2, 2, 2), dtype=np.int64)
    y = np.ones((2, 2, 2), dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 2D boolean arrays
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([[True, True], [False, False]], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 0D arrays (scalars)
    x = np.array(5.5, dtype=np.float32)
    y = np.array(5.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 2D Broadcasting (same number of dimensions)
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([[1, 2, 3]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 3D Broadcasting (same number of dimensions)
    x = np.array([[[1]], [[2]]], dtype=np.float32)  # shape (2, 1, 1)
    y = np.array([[[1, 2]], [[3, 4]]], dtype=np.float32)  # shape (2, 1, 2)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 2D uint8 arrays
    x = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    y = np.array([[0, 254], [128, 64]], dtype=np.uint8)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 1D float16 arrays
    x = np.array([1.5, -2.5, 0.0], dtype=np.float16)
    y = np.array([1.5, -2.5, 0.0], dtype=np.float16)
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: Large 4D float32 arrays
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    y = copy.deepcopy(x)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.lax.eq_1"] = jax_lax_eq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.eq_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.eq_1'.")


check_valid('jax.lax.eq', generated_inputs['jax.lax.eq_1'], lib="jax", suffix=1)
