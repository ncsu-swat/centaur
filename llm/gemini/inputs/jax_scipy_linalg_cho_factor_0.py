
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cho_factor_inputs():
    list_of_inputs = []

    def make_pd(shape, dtype):
        M = np.random.randn(*shape).astype(dtype)
        dims = list(range(len(shape)))
        dims[-1], dims[-2] = dims[-2], dims[-1]
        Mt = np.transpose(M, dims)
        A = np.matmul(M, Mt)
        N = shape[-1]
        eye = np.eye(N, dtype=dtype)
        for _ in range(len(shape) - 2):
            eye = np.expand_dims(eye, axis=0)
        A += eye * 0.1  # Add perturbation to ensure strict positive-definiteness
        return A

    # Input 1: 2x2 float32, lower=True
    a = make_pd((2, 2), np.float32)
    list_of_inputs.append({
        'a': a,
        'lower': True,
        'overwrite_a': False,
        'check_finite': True
    })

    # Input 2: 3x3 float64, lower=False
    a = make_pd((3, 3), np.float64)
    list_of_inputs.append({
        'a': a,
        'lower': False,
        'overwrite_a': True,
        'check_finite': False
    })

    # Input 3: Batched (2, 4, 4) float32, lower=True
    a = make_pd((2, 4, 4), np.float32)
    list_of_inputs.append({
        'a': a,
        'lower': True,
        'overwrite_a': True,
        'check_finite': True
    })

    # Input 4: Batched (3, 2, 5, 5) float64, lower=False
    a = make_pd((3, 2, 5, 5), np.float64)
    list_of_inputs.append({
        'a': a,
        'lower': False,
        'overwrite_a': False,
        'check_finite': False
    })

    # Input 5: 10x10 float32, lower=True
    a = make_pd((10, 10), np.float32)
    list_of_inputs.append({
        'a': a,
        'lower': True,
        'overwrite_a': False,
        'check_finite': False
    })

    # Input 6: 1x1 float64, lower=False
    a = make_pd((1, 1), np.float64)
    list_of_inputs.append({
        'a': a,
        'lower': False,
        'overwrite_a': True,
        'check_finite': True
    })

    # Input 7: Batched (4, 3, 3) float32, lower=False
    a = make_pd((4, 3, 3), np.float32)
    list_of_inputs.append({
        'a': a,
        'lower': False,
        'overwrite_a': False,
        'check_finite': True
    })

    # Input 8: 6x6 float64, lower=True
    a = make_pd((6, 6), np.float64)
    list_of_inputs.append({
        'a': a,
        'lower': True,
        'overwrite_a': True,
        'check_finite': True
    })

    # Input 9: Batched (1, 5, 5) float32, lower=True
    a = make_pd((1, 5, 5), np.float32)
    list_of_inputs.append({
        'a': a,
        'lower': True,
        'overwrite_a': False,
        'check_finite': True
    })

    # Input 10: 8x8 float32, lower=False
    a = make_pd((8, 8), np.float32)
    list_of_inputs.append({
        'a': a,
        'lower': False,
        'overwrite_a': False,
        'check_finite': False
    })

    return list_of_inputs

generated_inputs["jax.scipy.linalg.cho_factor"] = cho_factor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.cho_factor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.cho_factor'.")


check_valid('jax.scipy.linalg.cho_factor', generated_inputs['jax.scipy.linalg.cho_factor'], lib="jax", suffix=0)
