
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.scipy.linalg

try:
    import jax._src.scipy.linalg as jax_src_linalg
except ImportError:
    jax_src_linalg = None

original_eigh = jax.scipy.linalg.eigh

def patched_eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, turbo=True, eigvals=None, type=1, check_finite=True):
    return original_eigh(
        a, b=None, lower=lower, eigvals_only=eigvals_only,
        overwrite_a=overwrite_a, overwrite_b=overwrite_b,
        turbo=turbo, eigvals=None, type=type, check_finite=check_finite
    )

jax.scipy.linalg.eigh = patched_eigh
if jax_src_linalg is not None:
    jax_src_linalg.eigh = patched_eigh

def eigh_inputs():
    list_of_inputs = []

    # Input 1: 2x2 float32 symmetric matrix
    X = np.random.randn(2, 2).astype(np.float32)
    a = X + X.T
    b = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        'a': a,
        'b': b,
        'lower': True,
        'eigvals_only': False,
        'overwrite_a': False,
        'overwrite_b': False,
        'turbo': True,
        'eigvals': (0, 1),
        'type': 1,
        'check_finite': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 float64 symmetric matrix
    X = np.random.randn(3, 3).astype(np.float64)
    a = X + X.T
    b = np.random.randn(3, 3).astype(np.float64)
    input_dict = {
        'a': a,
        'b': b,
        'lower': False,
        'eigvals_only': True,
        'overwrite_a': True,
        'overwrite_b': True,
        'turbo': False,
        'eigvals': (1, 2),
        'type': 1,
        'check_finite': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4x4 complex64 Hermitian matrix
    X = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    a = X + X.conj().T
    b = np.random.randn(4, 4).astype(np.complex64)
    input_dict = {
        'a': a,
        'b': b,
        'lower': True,
        'eigvals_only': False,
        'overwrite_a': False,
        'overwrite_b': False,
        'turbo': True,
        'eigvals': (0, 3),
        'type': 1,
        'check_finite': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched 2x3x3 float32 symmetric matrix
    X = np.random.randn(2, 3, 3).astype(np.float32)
    a = X + X.transpose(0, 2, 1)
    b = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {
        'a': a,
        'b': b,
        'lower': False,
        'eigvals_only': False,
        'overwrite_a': False,
        'overwrite_b': False,
        'turbo': True,
        'eigvals': (0, 2),
        'type': 1,
        'check_finite': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 5x5 float32 symmetric matrix
    X = np.random.randn(5, 5).astype(np.float32)
    a = X + X.T
    b = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        'a': a,
        'b': b,
        'lower': True,
        'eigvals_only': True,
        'overwrite_a': False,
        'overwrite_b': False,
        'turbo': False,
        'eigvals': (0, 4),
        'type': 1,
        'check_finite': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2x2 complex128 Hermitian matrix
    X = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    a = X + X.conj().T
    b = np.random.randn(2, 2).astype(np.complex128)
    input_dict = {
        'a': a,
        'b': b,
        'lower': False,
        'eigvals_only': False,
        'overwrite_a': True,
        'overwrite_b': False,
        'turbo': True,
        'eigvals': (0, 1),
        'type': 1,
        'check_finite': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 8x8 float64 symmetric matrix
    X = np.random.randn(8, 8).astype(np.float64)
    a = X + X.T
    b = np.random.randn(8, 8).astype(np.float64)
    input_dict = {
        'a': a,
        'b': b,
        'lower': True,
        'eigvals_only': False,
        'overwrite_a': False,
        'overwrite_b': True,
        'turbo': True,
        'eigvals': (2, 5),
        'type': 1,
        'check_finite': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1x1 float32 symmetric matrix
    X = np.random.randn(1, 1).astype(np.float32)
    a = X + X.T
    b = np.random.randn(1, 1).astype(np.float32)
    input_dict = {
        'a': a,
        'b': b,
        'lower': True,
        'eigvals_only': True,
        'overwrite_a': False,
        'overwrite_b': False,
        'turbo': True,
        'eigvals': (0, 0),
        'type': 1,
        'check_finite': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batched 4x2x2 complex64 Hermitian matrix
    X = (np.random.randn(4, 2, 2) + 1j * np.random.randn(4, 2, 2)).astype(np.complex64)
    a = X + X.conj().transpose(0, 2, 1)
    b = np.random.randn(4, 2, 2).astype(np.complex64)
    input_dict = {
        'a': a,
        'b': b,
        'lower': True,
        'eigvals_only': False,
        'overwrite_a': False,
        'overwrite_b': False,
        'turbo': False,
        'eigvals': (0, 1),
        'type': 1,
        'check_finite': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6x6 float32 symmetric matrix
    X = np.random.randn(6, 6).astype(np.float32)
    a = X + X.T
    b = np.random.randn(6, 6).astype(np.float32)
    input_dict = {
        'a': a,
        'b': b,
        'lower': False,
        'eigvals_only': True,
        'overwrite_a': False,
        'overwrite_b': False,
        'turbo': True,
        'eigvals': (1, 4),
        'type': 1,
        'check_finite': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.eigh"] = eigh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.eigh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.eigh'.")


check_valid('jax.scipy.linalg.eigh', generated_inputs['jax.scipy.linalg.eigh'], lib="jax", suffix=0)
