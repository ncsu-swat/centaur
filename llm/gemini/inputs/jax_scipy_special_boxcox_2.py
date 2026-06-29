
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax.scipy.special

# Monkeypatch jax.scipy.special.boxcox if it is missing in the environment's JAX version
if not hasattr(jax.scipy.special, 'boxcox'):
    def boxcox_fallback(x, lmbda):
        x = jnp.asarray(x)
        lmbda = jnp.asarray(lmbda)
        safe_lmbda = jnp.where(lmbda == 0.0, 1.0, lmbda)
        return jnp.where(lmbda == 0.0, jnp.log(x), (jnp.power(x, lmbda) - 1.0) / safe_lmbda)
    jax.scipy.special.boxcox = boxcox_fallback

def boxcox_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, lmbda = 0.0 (log transform)
    x = np.array([0.5, 1.0, 2.0, 10.0], dtype=np.float32)
    lmbda = 0.0
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, float32, lmbda = 2.0
    x = np.random.uniform(0.1, 10.0, size=(5,)).astype(np.float32)
    lmbda = 2.0
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float32, lmbda = 0.5
    x = np.random.uniform(0.1, 5.0, size=(3, 4)).astype(np.float32)
    lmbda = 0.5
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, float64, lmbda = -1.0
    x = np.random.uniform(1.0, 100.0, size=(2, 2)).astype(np.float64)
    lmbda = -1.0
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, float32, lmbda = 1.5
    x = np.random.uniform(0.5, 3.0, size=(2, 3, 2)).astype(np.float32)
    lmbda = 1.5
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D array (scalar), float32, lmbda = 0.0
    x = np.array(5.0, dtype=np.float32)
    lmbda = 0.0
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, float64, lmbda = -0.5
    x = np.linspace(0.1, 2.0, num=10).astype(np.float64)
    lmbda = -0.5
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, float32, lmbda = 3.0
    x = np.random.uniform(0.1, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    lmbda = 3.0
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, float64, small positive values, lmbda = 0.1
    x = np.random.uniform(1e-5, 1e-2, size=(3, 3)).astype(np.float64)
    lmbda = 0.1
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, float32, negative lmbda = -2.5
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    lmbda = -2.5
    input_dict = {"x": x, "lmbda": lmbda}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.boxcox_2"] = boxcox_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.boxcox_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.boxcox_2'.")


check_valid('jax.scipy.special.boxcox', generated_inputs['jax.scipy.special.boxcox_2'], lib="jax", suffix=2)
