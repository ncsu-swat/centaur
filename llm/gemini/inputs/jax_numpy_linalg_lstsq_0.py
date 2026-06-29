
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lstsq_inputs():
    list_of_inputs = []

    # Input 1: Square matrix, 1D RHS, negative rcond, numpy_resid False
    a = np.random.randn(5, 5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": -1.0,
        "numpy_resid": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tall matrix, 1D RHS, positive rcond, numpy_resid True
    a = np.random.randn(10, 5).astype(np.float32)
    b = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": 1e-5,
        "numpy_resid": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Wide matrix, 2D RHS, numpy_resid False
    a = np.random.randn(4, 6).astype(np.float32)
    b = np.random.randn(4, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": 1e-3,
        "numpy_resid": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Square matrix, 1D RHS, float64, numpy_resid True
    a = np.random.randn(8, 8).astype(np.float64)
    b = np.random.randn(8).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": 1e-15,
        "numpy_resid": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tall matrix, 2D RHS, float64, numpy_resid False
    a = np.random.randn(15, 10).astype(np.float64)
    b = np.random.randn(15, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": 1e-8,
        "numpy_resid": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small square matrix, 1D RHS, zero rcond, numpy_resid True
    a = np.random.randn(2, 2).astype(np.float32)
    b = np.random.randn(2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": 0.0,
        "numpy_resid": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tall matrix, 1D RHS, float32, numpy_resid False
    a = np.random.randn(20, 10).astype(np.float32)
    b = np.random.randn(20).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": 1e-6,
        "numpy_resid": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Wide matrix, 2D RHS, negative float64 rcond, numpy_resid True
    a = np.random.randn(5, 10).astype(np.float64)
    b = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": -0.5,
        "numpy_resid": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large square matrix, 1D RHS, numpy_resid False
    a = np.random.randn(50, 50).astype(np.float32)
    b = np.random.randn(50).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": 1e-4,
        "numpy_resid": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Medium tall matrix, 2D RHS, float64, numpy_resid True
    a = np.random.randn(12, 6).astype(np.float64)
    b = np.random.randn(12, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "rcond": 0.001,
        "numpy_resid": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.lstsq"] = lstsq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.lstsq'.")


check_valid('jax.numpy.linalg.lstsq', generated_inputs['jax.numpy.linalg.lstsq'], lib="jax", suffix=0)
