
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanquantile_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, axis 1, standard weights
    a = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 6.0]], dtype=np.float32)
    q = np.array([0.5], dtype=np.float32)
    axis = 1
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = False
    weights = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, negative values, axis 0, keepdims True
    a = np.array([[-1.0, -2.0], [np.nan, -4.0]], dtype=np.float32)
    q = np.array([0.25, 0.75], dtype=np.float32)
    axis = 0
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = True
    weights = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, float64, axis 2, multiple quantiles
    a = np.array([[[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]], 
                  [[7.0, np.nan], [9.0, 10.0], [11.0, 12.0]]], dtype=np.float64)
    q = np.array([0.1, 0.9], dtype=np.float64)
    axis = 2
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = False
    weights = np.ones_like(a, dtype=np.float64)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, float32, axis 0
    a = np.array([1.5, np.nan, 2.5, 3.5, np.nan], dtype=np.float32)
    q = np.array([0.5], dtype=np.float32)
    axis = 0
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = False
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, float32, negative axis, keepdims True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[0, 1, 0, 1] = np.nan
    q = np.array([0.0, 1.0], dtype=np.float32)
    axis = -1
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = True
    weights = np.ones_like(a, dtype=np.float32)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, negative axis, float32
    a = np.array([[10.0, np.nan, 30.0], [40.0, 50.0, np.nan]], dtype=np.float32)
    q = np.array([0.3], dtype=np.float32)
    axis = -2
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = False
    weights = np.ones_like(a, dtype=np.float32)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with dimension size 1, axis 0
    a = np.array([[[1.0]], [[np.nan]], [[3.0]], [[4.0]]], dtype=np.float32)
    q = np.array([0.5], dtype=np.float32)
    axis = 0
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = True
    weights = np.ones_like(a, dtype=np.float32)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, float64, axis 1, varying weights
    a = np.array([[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, np.nan]], dtype=np.float64)
    q = np.array([0.8], dtype=np.float64)
    axis = 1
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = False
    weights = np.array([[1.0, 1.0, 1.0], [2.0, 0.5, 1.5], [1.0, 1.0, 1.0]], dtype=np.float64)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, float32, no NaNs
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], dtype=np.float32)
    q = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    axis = 0
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = True
    weights = np.ones_like(a, dtype=np.float32)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, float32, axis 1, keepdims False
    a = np.array([[[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], 
                  [[np.nan, 8.0, 9.0], [10.0, np.nan, 12.0]]], dtype=np.float32)
    q = np.array([0.5], dtype=np.float32)
    axis = 1
    overwrite_input = False
    method = 'inverted_cdf'
    keepdims = False
    weights = np.ones_like(a, dtype=np.float32)
    
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanquantile_1"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_1'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_1'], lib="jax", suffix=1)
