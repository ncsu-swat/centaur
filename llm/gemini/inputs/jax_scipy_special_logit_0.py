
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logit_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 in (0, 1)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array float32 with values in (0, 1)
    x = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array float64
    x = np.linspace(0.01, 0.99, 10, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array float32
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array float16
    x = np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.2]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array
    x = np.random.uniform(0.05, 0.95, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array
    x = np.random.uniform(0.1, 0.9, size=(1, 2, 2, 3)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Boundary values 0.0 and 1.0
    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Values outside (0, 1) including negatives
    x = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Integer array
    x = np.array([0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.logit"] = logit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.logit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.logit'.")


check_valid('jax.scipy.special.logit', generated_inputs['jax.scipy.special.logit'], lib="jax", suffix=0)
