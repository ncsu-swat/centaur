
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gammaln_inputs():
    list_of_inputs = []

    # All inputs use the same shape (4,) and dtype float32 to minimize JAX compilation time.
    list_of_inputs.append({"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([1.2, 1.8, 2.4, 3.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([-0.1, -1.1, -2.1, -3.1], dtype=np.float32)})

    return list_of_inputs

generated_inputs["jax.scipy.special.gammaln"] = gammaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.gammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.gammaln'.")


check_valid('jax.scipy.special.gammaln', generated_inputs['jax.scipy.special.gammaln'], lib="jax", suffix=0)
