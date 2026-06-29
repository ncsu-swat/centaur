
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeta_inputs():
    list_of_inputs = []

    # Input 1: Float32 numpy scalars
    list_of_inputs.append({
        "x": np.float32(2.0),
        "q": np.float32(1.0)
    })

    # Input 2: Float64 numpy scalars
    list_of_inputs.append({
        "x": np.float64(3.0),
        "q": np.float64(0.5)
    })

    # Input 3: Standard python floats
    list_of_inputs.append({
        "x": 1.5,
        "q": 2.0
    })

    # Input 4: Large q float
    list_of_inputs.append({
        "x": 4.0,
        "q": 10.0
    })

    # Input 5: Small q float
    list_of_inputs.append({
        "x": np.float64(2.5),
        "q": np.float64(0.1)
    })

    # Input 6: Near 1 x float
    list_of_inputs.append({
        "x": np.float32(1.1),
        "q": np.float32(1.5)
    })

    # Input 7: High x float
    list_of_inputs.append({
        "x": 10.0,
        "q": 1.0
    })

    # Input 8: High q float
    list_of_inputs.append({
        "x": 2.0,
        "q": 100.0
    })

    # Input 9: Small fractional x and q
    list_of_inputs.append({
        "x": np.float64(1.05),
        "q": np.float64(0.01)
    })

    # Input 10: Float32 scalars
    list_of_inputs.append({
        "x": np.float32(5.5),
        "q": np.float32(2.5)
    })

    return list_of_inputs

generated_inputs["jax.lax.zeta_2"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.zeta_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.zeta_2'.")


check_valid('jax.lax.zeta', generated_inputs['jax.lax.zeta_2'], lib="jax", suffix=2)
