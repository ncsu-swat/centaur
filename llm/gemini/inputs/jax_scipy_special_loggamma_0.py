
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def loggamma_inputs():
    list_of_inputs = []

    # All inputs use the same shape (3,) and dtype float32 to minimize JAX compilation overhead.
    list_of_inputs.append({"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.5, 1.5, 2.5], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([4.0, 5.0, 6.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.1, 0.2, 0.3], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([10.0, 11.0, 12.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.75, 1.25, 1.75], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([2.2, 3.3, 4.4], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([5.5, 6.6, 7.7], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([8.8, 9.9, 10.1], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.01, 0.05, 0.1], dtype=np.float32)})

    return list_of_inputs

generated_inputs["jax.scipy.special.loggamma"] = loggamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.loggamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.loggamma'.")


check_valid('jax.scipy.special.loggamma', generated_inputs['jax.scipy.special.loggamma'], lib="jax", suffix=0)
