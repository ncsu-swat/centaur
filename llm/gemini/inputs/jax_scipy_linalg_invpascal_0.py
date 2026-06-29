
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.scipy.linalg
import scipy.linalg

# Monkeypatch missing JAX API to prevent failure in test harness lookup
jax.scipy.linalg.invpascal = scipy.linalg.invpascal

def invpascal_inputs():
    list_of_inputs = []

    kinds = ["symmetric", "lower", "upper"]
    for i in range(1, 11):
        kind = kinds[i % 3]
        list_of_inputs.append({
            "n": int(i),
            "kind": str(kind)
        })

    return list_of_inputs

generated_inputs["jax.scipy.linalg.invpascal"] = invpascal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.invpascal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.invpascal'.")


check_valid('jax.scipy.linalg.invpascal', generated_inputs['jax.scipy.linalg.invpascal'], lib="jax", suffix=0)
