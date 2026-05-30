
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hypot_inputs():
    list_of_inputs = []

    # Case 1: 1D float32 arrays, same size, positive values
    x1 = np.array([3.0, 5.0, 8.0], dtype=np.float32)
    x2 = np.array([4.0, 12.0, 15.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 2: 1D float32 arrays, same size, negative values
    x1 = np.array([-3.0, -5.0, -8.0], dtype=np.float32)
    x2 = np.array([-4.0, -12.0, -15.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 3: 2D float64 arrays, same size, mixed values
    x1 = np.random.uniform(-10, 10, (3, 4)).astype(np.float64)
    x2 = np.random.uniform(-10, 10, (3, 4)).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 4: Broadcasting: (1, 3) and (3, 1) float32 arrays
    x1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x2 = np.array([[4.0], [5.0], [6.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 5: 0D arrays (scalars as arrays)
    x1 = np.array(3.0, dtype=np.float32)
    x2 = np.array(4.0, dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 6: 3D float32 arrays, same size
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 7: Integer arrays (int32)
    x1 = np.array([3, 5, 8], dtype=np.int32)
    x2 = np.array([4, 12, 15], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 8: Integer arrays (int64) with broadcasting
    x1 = np.arange(5, dtype=np.int64).reshape(5, 1)
    x2 = np.arange(5, dtype=np.int64).reshape(1, 5)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 9: 4D float64 arrays, same size
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float64)
    x2 = np.random.randn(2, 2, 3, 3).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Case 10: Large values (to verify overflow/underflow handling)
    x1 = np.array([1e150, 1e-150], dtype=np.float64)
    x2 = np.array([1e150, 1e-150], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.hypot_1"] = hypot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hypot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hypot_1'.")


check_valid('jax.numpy.hypot', generated_inputs['jax.numpy.hypot_1'], lib="jax", suffix=1)
