
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def clip_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 array, scalar min and max
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    min_val = np.array(2, dtype=np.int32)
    max_val = np.array(4, dtype=np.int32)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 2: Float32 1D array with negative values
    arr = np.array([-2.5, -1.0, 0.0, 1.5, 2.5], dtype=np.float32)
    min_val = np.array(-1.5, dtype=np.float32)
    max_val = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 3: 2D array, broadcasting 1D min and max
    arr = np.random.randn(3, 4).astype(np.float32)
    min_val = np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32)
    max_val = np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 4: 3D array, scalar min and max (float64)
    arr = np.random.randn(2, 3, 4).astype(np.float64)
    min_val = np.array(-0.5, dtype=np.float64)
    max_val = np.array(0.5, dtype=np.float64)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 5: Int64 arrays, matching shapes
    arr = np.array([[10, 20], [30, 40]], dtype=np.int64)
    min_val = np.array([[12, 18], [32, 38]], dtype=np.int64)
    max_val = np.array([[15, 25], [35, 45]], dtype=np.int64)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 6: Float16 1D array
    arr = np.array([-10.0, 0.0, 10.0], dtype=np.float16)
    min_val = np.array(-5.0, dtype=np.float16)
    max_val = np.array(5.0, dtype=np.float16)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 7: 4D array with 4D broadcastable min and max
    arr = np.random.randn(2, 2, 2, 2).astype(np.float32)
    min_val = np.array([[[[-1.0]]]], dtype=np.float32)
    max_val = np.array([[[[1.0]]]], dtype=np.float32)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 8: 2D array where min is larger than max
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    min_val = np.array([[5, 5], [5, 5]], dtype=np.int32)
    max_val = np.array([[2, 2], [2, 2]], dtype=np.int32)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 9: Large 1D array of integers
    arr = np.arange(-100, 100).astype(np.int32)
    min_val = np.array(-50, dtype=np.int32)
    max_val = np.array(50, dtype=np.int32)
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    # Input 10: Multi-dimensional broadcasting float64 arrays
    arr = np.ones((5, 1, 5), dtype=np.float64)
    min_val = np.zeros((1, 5, 1), dtype=np.float64)
    max_val = np.ones((1, 5, 1), dtype=np.float64) * 2.0
    list_of_inputs.append({"arr": arr, "min": min_val, "max": max_val})

    return list_of_inputs

generated_inputs["jax.numpy.clip_1"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.clip_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.clip_1'.")


check_valid('jax.numpy.clip', generated_inputs['jax.numpy.clip_1'], lib="jax", suffix=1)
