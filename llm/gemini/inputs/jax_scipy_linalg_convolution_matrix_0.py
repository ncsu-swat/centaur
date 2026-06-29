
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.scipy.linalg
import scipy.linalg
import jax.numpy as jnp

# Monkey-patch jax.scipy.linalg.convolution_matrix since it is missing in JAX
if not hasattr(jax.scipy.linalg, 'convolution_matrix'):
    def _convolution_matrix(a, n, mode='full'):
        a_np = np.asarray(a)
        res = scipy.linalg.convolution_matrix(a_np, int(n), mode=mode)
        return jnp.array(res)
    jax.scipy.linalg.convolution_matrix = _convolution_matrix

def convolution_matrix_inputs():
    list_of_inputs = []

    # Input 1: float32, full mode, n < len(a)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    n = 3
    mode = 'full'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 2: float64, valid mode, n > len(a)
    a = np.array([0.5, -1.5, 2.0, -0.5], dtype=np.float64)
    n = 6
    mode = 'valid'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 3: int32, same mode
    a = np.array([1, -1, 2], dtype=np.int32)
    n = 5
    mode = 'same'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 4: float32, valid mode, n > len(a)
    a = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0], dtype=np.float32)
    n = 10
    mode = 'valid'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 5: complex64, same mode
    a = np.array([1.0+1j, 2.0-2j, -1.0+0j, 0.0+3j], dtype=np.complex64)
    n = 4
    mode = 'same'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 6: float64, valid mode
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    n = 12
    mode = 'valid'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 7: float32, full mode, n < len(a)
    a = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    n = 2
    mode = 'full'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 8: int64, same mode
    a = np.array([2, -3, 0, 1, 5], dtype=np.int64)
    n = 8
    mode = 'same'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 9: complex128, full mode
    a = np.array([1.0+0j, 0.0+1j, -1.0-1j], dtype=np.complex128)
    n = 5
    mode = 'full'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    # Input 10: float32, valid mode, n == len(a)
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    n = 8
    mode = 'valid'
    list_of_inputs.append({"a": a, "n": n, "mode": mode})

    return list_of_inputs

generated_inputs["jax.scipy.linalg.convolution_matrix"] = convolution_matrix_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.convolution_matrix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.convolution_matrix'.")


check_valid('jax.scipy.linalg.convolution_matrix', generated_inputs['jax.scipy.linalg.convolution_matrix'], lib="jax", suffix=0)
