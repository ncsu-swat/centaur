
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class CallableTensorMock:
    def __init__(self, func):
        self._func = func
        self.shape = ()
        self.ndim = 0
        self.dtype = np.dtype('float32')

    def __call__(self, x):
        return self._func(x)

def funm_inputs():
    list_of_inputs = []

    def gen_pd(n, dtype):
        B = np.random.randn(n, n).astype(dtype)
        return B @ B.T + np.eye(n).astype(dtype)

    # Input 1: float32, 2x2, np.sin, disp=True
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.sin),
        "disp": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 3x3, np.cos, disp=False
    A = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.cos),
        "disp": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 4x4, np.sinh, disp=True
    A = np.random.randn(4, 4).astype(np.float64)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.sinh),
        "disp": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, 2x2, np.cosh, disp=False
    A = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.cosh),
        "disp": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 Positive Definite, 3x3, np.log, disp=True
    A = gen_pd(3, np.float32)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.log),
        "disp": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 Positive Definite, 2x2, np.sqrt, disp=False
    A = gen_pd(2, np.float64)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.sqrt),
        "disp": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 5x5, np.square, disp=True
    A = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.square),
        "disp": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128, 3x3, np.exp, disp=False
    A = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex128)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.exp),
        "disp": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, 2x2, np.arctan, disp=True
    A = np.random.randn(2, 2).astype(np.float64)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.arctan),
        "disp": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 diagonal matrix, 4x4, np.tanh, disp=False
    A = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    input_dict = {
        "A": A,
        "func": CallableTensorMock(np.tanh),
        "disp": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.funm"] = funm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.funm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.funm'.")


check_valid('jax.scipy.linalg.funm', generated_inputs['jax.scipy.linalg.funm'], lib="jax", suffix=0)
