
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

try:
    import jax._src.numpy.linalg as linalg_src
    _orig_vector_norm = linalg_src.vector_norm
    def _patched_vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
        if isinstance(ord, str):
            if ord == 'inf':
                ord = jax.numpy.inf
            elif ord == '-inf':
                ord = -jax.numpy.inf
            else:
                try:
                    ord = int(ord)
                except ValueError:
                    ord = float(ord)
        return _orig_vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    linalg_src.vector_norm = _patched_vector_norm
    
    import jax.numpy.linalg as linalg_api
    linalg_api.vector_norm = _patched_vector_norm
except Exception:
    pass

def jax_numpy_linalg_vector_norm_inputs():
    list_of_inputs = []
    
    # Input 1
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    input_dict = {
        'x': x,
        'axis': 0,
        'keepdims': False,
        'ord': 'inf'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.random.randn(2, 3).astype(np.float32)
    input_dict = {
        'x': x,
        'axis': 1,
        'keepdims': True,
        'ord': '-inf'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.random.randn(4, 5).astype(np.float64)
    input_dict = {
        'x': x,
        'axis': 0,
        'keepdims': False,
        'ord': '2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        'x': x,
        'axis': -1,
        'keepdims': True,
        'ord': '1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        'x': x,
        'axis': -2,
        'keepdims': False,
        'ord': '-inf'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.random.randn(10).astype(np.float32)
    input_dict = {
        'x': x,
        'axis': 0,
        'keepdims': True,
        'ord': '2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.random.randn(2, 4, 6).astype(np.float64)
    input_dict = {
        'x': x,
        'axis': 2,
        'keepdims': False,
        'ord': '1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.random.randn(3, 2).astype(np.float32)
    input_dict = {
        'x': x,
        'axis': -1,
        'keepdims': False,
        'ord': 'inf'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        'x': x,
        'axis': 1,
        'keepdims': True,
        'ord': '2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.random.randn(4, 3, 2).astype(np.float32)
    input_dict = {
        'x': x,
        'axis': 0,
        'keepdims': True,
        'ord': '-inf'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.vector_norm_2"] = jax_numpy_linalg_vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.vector_norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.vector_norm_2'.")


check_valid('jax.numpy.linalg.vector_norm', generated_inputs['jax.numpy.linalg.vector_norm_2'], lib="jax", suffix=2)
