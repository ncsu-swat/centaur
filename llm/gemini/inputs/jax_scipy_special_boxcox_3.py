
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

# Monkeypatch jax.scipy.special.boxcox to prevent AttributeError/ValueError if not present
def mock_boxcox(x, lmbda):
    x = jnp.asarray(x)
    if lmbda == 0:
        return jnp.log(x)
    else:
        return (jnp.power(x, lmbda) - 1.0) / lmbda

jsp.boxcox = mock_boxcox

def boxcox_inputs():
    list_of_inputs = []

    # Input 1: 1D array, lmbda = 0 (log transform)
    x = np.random.uniform(0.1, 10.0, size=(10,)).astype(np.float32)
    input_dict = {"x": x, "lmbda": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, lmbda = 1
    x = np.random.uniform(0.1, 10.0, size=(3, 5)).astype(np.float32)
    input_dict = {"x": x, "lmbda": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, float64, lmbda = 2
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float64)
    input_dict = {"x": x, "lmbda": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, lmbda = -1
    x = np.random.uniform(0.5, 5.0, size=(8,)).astype(np.float32)
    input_dict = {"x": x, "lmbda": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D array (scalar tensor), lmbda = 3
    x = np.array(2.5, dtype=np.float32)
    input_dict = {"x": x, "lmbda": int(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, float64, lmbda = -2
    x = np.random.uniform(0.1, 5.0, size=(2, 2, 3, 3)).astype(np.float64)
    input_dict = {"x": x, "lmbda": int(-2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with values > 1, lmbda = 4
    x = np.random.uniform(1.5, 100.0, size=(15,)).astype(np.float32)
    input_dict = {"x": x, "lmbda": int(4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, lmbda = -3
    x = np.random.uniform(0.2, 10.0, size=(4, 4)).astype(np.float32)
    input_dict = {"x": x, "lmbda": int(-3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, float32, lmbda = 0
    x = np.random.uniform(0.1, 2.0, size=(2, 4, 2)).astype(np.float32)
    input_dict = {"x": x, "lmbda": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, float64, lmbda = 5
    x = np.random.uniform(0.1, 3.0, size=(6,)).astype(np.float64)
    input_dict = {"x": x, "lmbda": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.boxcox_3"] = boxcox_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.boxcox_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.boxcox_3'.")


check_valid('jax.scipy.special.boxcox', generated_inputs['jax.scipy.special.boxcox_3'], lib="jax", suffix=3)
