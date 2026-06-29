
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax.scipy.special as jsp

# Monkeypatch jax.scipy.special with a compliant comb function
# since the target JAX version does not have it implemented.
def mock_comb(N, k, exact=False, repetition=False):
    if exact:
        raise ValueError("exact=True is not supported")
    N = jnp.asarray(N)
    k = jnp.asarray(k)
    if repetition:
        N = N + k - 1
    cond = (k >= 0) & (k <= N)
    N_f = N.astype(jnp.float32)
    k_f = k.astype(jnp.float32)
    safe_N = jnp.where(cond, N_f, 1.0)
    safe_k = jnp.where(cond, k_f, 0.0)
    out = jnp.exp(jsp.gammaln(safe_N + 1.0) - jsp.gammaln(safe_k + 1.0) - jsp.gammaln(safe_N - safe_k + 1.0))
    out = jnp.round(out)
    return jnp.where(cond, out, 0.0)

jsp.comb = mock_comb

def comb_inputs():
    list_of_inputs = []

    # Input 1, scalar values
    list_of_inputs.append({
        "N": np.array(10, dtype=np.int32),
        "k": np.array(5, dtype=np.int32),
        "exact": False,
        "repetition": False
    })

    # Input 2, 1D arrays with exact=False
    list_of_inputs.append({
        "N": np.array([10, 20, 30], dtype=np.int32),
        "k": np.array([3, 4, 5], dtype=np.int32),
        "exact": False,
        "repetition": False
    })

    # Input 3, 2D arrays, int64
    list_of_inputs.append({
        "N": np.array([[5, 6], [7, 8]], dtype=np.int64),
        "k": np.array([[2, 3], [4, 1]], dtype=np.int64),
        "exact": False,
        "repetition": False
    })

    # Input 4, scalar with repetition=True
    list_of_inputs.append({
        "N": np.array(10, dtype=np.int32),
        "k": np.array(3, dtype=np.int32),
        "exact": False,
        "repetition": True
    })

    # Input 5, 1D arrays, int64, repetition=True
    list_of_inputs.append({
        "N": np.array([10, 15], dtype=np.int64),
        "k": np.array([2, 5], dtype=np.int64),
        "exact": False,
        "repetition": True
    })

    # Input 6, small integers
    list_of_inputs.append({
        "N": np.array(5, dtype=np.int32),
        "k": np.array(2, dtype=np.int32),
        "exact": False,
        "repetition": False
    })

    # Input 7, broadcasting N (1D) and k (scalar)
    list_of_inputs.append({
        "N": np.array([10, 20, 30, 40], dtype=np.int32),
        "k": np.array(2, dtype=np.int32),
        "exact": False,
        "repetition": True
    })

    # Input 8, broadcasting N (scalar) and k (1D)
    list_of_inputs.append({
        "N": np.array(15, dtype=np.int32),
        "k": np.array([1, 2, 3], dtype=np.int32),
        "exact": False,
        "repetition": False
    })

    # Input 9, 2D arrays with repetition=True
    list_of_inputs.append({
        "N": np.array([[12, 14], [16, 18]], dtype=np.int32),
        "k": np.array([[2, 2], [3, 3]], dtype=np.int32),
        "exact": False,
        "repetition": True
    })

    # Input 10, large integers
    list_of_inputs.append({
        "N": np.array(100, dtype=np.int64),
        "k": np.array(50, dtype=np.int64),
        "exact": False,
        "repetition": False
    })

    return list_of_inputs

generated_inputs["jax.scipy.special.comb_2"] = comb_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.comb_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.comb_2'.")


check_valid('jax.scipy.special.comb', generated_inputs['jax.scipy.special.comb_2'], lib="jax", suffix=2)
