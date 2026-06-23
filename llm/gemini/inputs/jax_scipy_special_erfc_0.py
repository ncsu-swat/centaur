
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

def erfc_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({"x": np.array([1.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([-1.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.5], dtype=np.float64)})
    list_of_inputs.append({"x": np.array([[1.0]], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([[[1.0]]], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([2.0, 3.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([-2.0, -3.0], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([0.1], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([-0.1], dtype=np.float32)})
    
    return list_of_inputs

generated_inputs["jax.scipy.special.erfc"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.erfc'.")


check_valid('jax.scipy.special.erfc', generated_inputs['jax.scipy.special.erfc'], lib="jax", suffix=0)
