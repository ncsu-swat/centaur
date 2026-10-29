
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gammasgn_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 positive
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar float64 negative non-integer
    x = np.array(-1.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with positive values
    x = np.array([0.5, 1.0, 2.0, 3.5, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array with negative non-integer values
    x = np.array([-0.5, -1.2, -2.7, -3.1], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array with special values (+0.0, -0.0, inf, -inf, nan)
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D array with float32
    x = np.linspace(-4.5, 4.5, 12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D array with float64
    x = np.array(
        [[-2.3, -1.8, -0.1], [0.1, 1.8, 2.3]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 3D array with various signs
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D array with float64
    x = np.random.uniform(0.1, 10.0, size=(2, 2, 2, 3)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Negative integer values where NaN is expected
    x = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.gammasgn"] = gammasgn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.gammasgn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.gammasgn'.")


check_valid('jax.scipy.special.gammasgn', generated_inputs['jax.scipy.special.gammasgn'], lib="jax", suffix=0)
