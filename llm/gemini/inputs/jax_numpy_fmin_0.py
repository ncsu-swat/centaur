
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmin_inputs():
    list_of_inputs = []
    
    # Input 1: Float32 1D arrays, same shape
    x1 = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    x2 = np.array([2.0, 1.0, -0.5], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 2: Float32 2D arrays, broadcast compatible
    x1 = np.random.randn(2, 3).astype(np.float32)
    x2 = np.random.randn(1, 3).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 3: Int32 arrays, 3D shape
    x1 = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 4: Float64 arrays with NaNs and Infs (edge cases)
    x1 = np.array([np.nan, 1.0, -np.inf, np.inf], dtype=np.float64)
    x2 = np.array([2.0, np.nan, 3.0, np.nan], dtype=np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 5: Float32 4D arrays, same shape
    x1 = np.random.randn(2, 3, 2, 2).astype(np.float32)
    x2 = np.random.randn(2, 3, 2, 2).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 6: Int64 2D arrays with negative values
    x1 = np.random.randint(-100, 100, size=(4, 4)).astype(np.int64)
    x2 = np.random.randint(-100, 100, size=(4, 4)).astype(np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 7: Float16 1D arrays
    x1 = np.array([0.5, -0.5, 12.0], dtype=np.float16)
    x2 = np.array([-1.5, 0.5, 5.0], dtype=np.float16)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 8: Broadcasting a 1-element array
    x1 = np.array([5.0], dtype=np.float32)
    x2 = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 9: Extreme large and small values in float64
    x1 = np.array([1e15, -1e15, 0.0], dtype=np.float64)
    x2 = np.array([-1e15, 1e15, -0.0], dtype=np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})
    
    # Input 10: Broadcasting (5, 1) and (1, 5) shapes
    x1 = np.random.randn(5, 1).astype(np.float32)
    x2 = np.random.randn(1, 5).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.fmin"] = fmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmin'.")


check_valid('jax.numpy.fmin', generated_inputs['jax.numpy.fmin'], lib="jax", suffix=0)
