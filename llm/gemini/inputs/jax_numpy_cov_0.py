
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cov_inputs():
    list_of_inputs = []

    # Input 1: rowvar=True, float32 2D arrays, ddof=1
    N = 10
    m = np.random.randn(3, N).astype(np.float32)
    y = np.random.randn(2, N).astype(np.float32)
    fweights = np.random.randint(1, 5, size=(N,), dtype=np.int32)
    aweights = np.random.uniform(0.5, 2.0, size=(N,)).astype(np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": False,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: rowvar=False, float64 2D arrays, bias=True, ddof=0
    N = 8
    m = np.random.randn(N, 4).astype(np.float64)
    y = np.random.randn(N, 2).astype(np.float64)
    fweights = np.random.randint(1, 4, size=(N,), dtype=np.int64)
    aweights = np.random.uniform(0.1, 1.0, size=(N,)).astype(np.float64)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": True,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: rowvar=True, 1D float32 arrays, ddof=0
    N = 12
    m = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    fweights = np.ones(N, dtype=np.int32)
    aweights = np.random.uniform(0.2, 1.5, size=(N,)).astype(np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": False,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: rowvar=False, 1D float64 arrays, ddof=1
    N = 15
    m = np.random.randn(N).astype(np.float64)
    y = np.random.randn(N).astype(np.float64)
    fweights = np.random.randint(1, 3, size=(N,), dtype=np.int64)
    aweights = np.ones(N, dtype=np.float64)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": False,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 inputs, rowvar=True
    N = 16
    m = (np.random.randn(4, N) + 1j * np.random.randn(4, N)).astype(np.complex64)
    y = (np.random.randn(1, N) + 1j * np.random.randn(1, N)).astype(np.complex64)
    fweights = np.random.randint(1, 4, size=(N,), dtype=np.int32)
    aweights = np.random.uniform(0.5, 1.5, size=(N,)).astype(np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": False,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128 inputs, rowvar=False, bias=True, ddof=0
    N = 20
    m = (np.random.randn(N, 3) + 1j * np.random.randn(N, 3)).astype(np.complex128)
    y = (np.random.randn(N, 2) + 1j * np.random.randn(N, 2)).astype(np.complex128)
    fweights = np.random.randint(1, 5, size=(N,), dtype=np.int64)
    aweights = np.random.uniform(0.1, 2.0, size=(N,)).astype(np.float64)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": True,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D m and 1D y, rowvar=True with negative values
    N = 6
    m = np.array([[-3.0, -1.0, 0.0, 1.5, 2.0, -0.5],
                  [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]], dtype=np.float32)
    y = np.array([-1.0, 0.0, 1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    fweights = np.array([1, 2, 1, 3, 1, 2], dtype=np.int32)
    aweights = np.array([0.5, 1.0, 1.5, 0.5, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": True,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: rowvar=False, 2D m with 1 variable, 2D y with 2 variables
    N = 7
    m = np.random.randn(N, 1).astype(np.float32)
    y = np.random.randn(N, 2).astype(np.float32)
    fweights = np.random.randint(1, 3, size=(N,), dtype=np.int32)
    aweights = np.random.uniform(0.8, 1.2, size=(N,)).astype(np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": False,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: larger dimensions, rowvar=True, float64
    N = 25
    m = np.random.randn(5, N).astype(np.float64)
    y = np.random.randn(3, N).astype(np.float64)
    fweights = np.random.randint(1, 4, size=(N,), dtype=np.int64)
    aweights = np.random.uniform(0.5, 1.5, size=(N,)).astype(np.float64)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": False,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: input float32 upcast to float64 via dtype parameter
    N = 10
    m = np.random.randn(N, 3).astype(np.float32)
    y = np.random.randn(N, 1).astype(np.float32)
    fweights = np.ones(N, dtype=np.int32)
    aweights = np.ones(N, dtype=np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": False,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.cov"] = cov_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cov' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cov'.")


check_valid('jax.numpy.cov', generated_inputs['jax.numpy.cov'], lib="jax", suffix=0)
