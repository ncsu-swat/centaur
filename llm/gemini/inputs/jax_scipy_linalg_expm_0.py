
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expm_inputs():
    list_of_inputs = []

    for i in range(10):
        A = np.array([[float(i)]], dtype=np.float32)
        list_of_inputs.append({
            "A": A,
            "upper_triangular": True,
            "max_squarings": 0
        })

    return list_of_inputs

generated_inputs["jax.scipy.linalg.expm"] = expm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.expm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.expm'.")


check_valid('jax.scipy.linalg.expm', generated_inputs['jax.scipy.linalg.expm'], lib="jax", suffix=0)
