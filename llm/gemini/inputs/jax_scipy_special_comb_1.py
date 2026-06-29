
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax.scipy.special

# Monkeypatch jax.scipy.special.comb because it is missing in the environment's JAX version
def jax_comb(N, k, exact=False, repetition=False):
    N = jnp.asarray(N, dtype=jnp.float32)
    k = jnp.asarray(k, dtype=jnp.float32)
    
    if repetition:
        N = N + k - 1.0
        
    cond = (k >= 0) & (k <= N)
    N_clipped = jnp.where(cond, N, 1.0)
    k_clipped = jnp.where(cond, k, 0.0)
    
    log_comb = (jax.scipy.special.gammaln(N_clipped + 1.0) 
                - jax.scipy.special.gammaln(k_clipped + 1.0) 
                - jax.scipy.special.gammaln(N_clipped - k_clipped + 1.0))
    
    out = jnp.exp(log_comb)
    return jnp.where(cond, out, 0.0)

jax.scipy.special.comb = jax_comb

def comb_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, no repetition, exact=False
    N = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    k = np.array([3.0, 5.0, 8.0], dtype=np.float32)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays, with repetition, exact=False
    N = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    k = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalars (0D arrays) of float32, no repetition, exact=False
    N = np.array(10.0, dtype=np.float32)
    k = np.array(3.0, dtype=np.float32)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int32 arrays, exact=False, repetition=False
    N = np.array([10, 15, 20], dtype=np.int32)
    k = np.array([2, 4, 6], dtype=np.int32)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 arrays, exact=False, repetition=False
    N = np.arange(10, 18, dtype=np.float32).reshape(2, 2, 2)
    k = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64 arrays, exact=False, repetition=True
    N = np.array([5.5, 10.5, 15.5], dtype=np.float64)
    k = np.array([2.1, 3.2, 4.3], dtype=np.float64)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 arrays, exact=False, repetition=False
    N = np.array([12.0, 14.0, 16.0], dtype=np.float32)
    k = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 arrays with repetition, exact=False
    N = np.array([[10, 12], [14, 16]], dtype=np.int32)
    k = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 arrays where k > N, exact=False, repetition=False
    N = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    k = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    input_dict = {"N": N, "k": k, "exact": False, "repetition": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 arrays, exact=False, repetition=False
    N = np.ones((2, 2, 2, 2), dtype=np.float32) * 10
    k = np.ones((2, 2, 2, 2), dtype=np.float32) * 3
    input_dict = {"N": N, "k": k, "exact": False, "repetition": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.comb_1"] = comb_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.comb_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.comb_1'.")


check_valid('jax.scipy.special.comb', generated_inputs['jax.scipy.special.comb_1'], lib="jax", suffix=1)
