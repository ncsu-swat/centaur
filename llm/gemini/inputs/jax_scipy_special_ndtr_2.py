
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ndtr_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive float
    input_dict = {"x": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float
    input_dict = {"x": -1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive float
    input_dict = {"x": 8.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative float
    input_dict = {"x": -8.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small positive float
    input_dict = {"x": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small negative float
    input_dict = {"x": -1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.float32 scalar positive
    input_dict = {"x": np.float32(2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.float32 scalar negative
    input_dict = {"x": np.float32(-2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.float64 scalar positive
    input_dict = {"x": np.float64(5.4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: np.float64 scalar negative
    input_dict = {"x": np.float64(-5.4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.ndtr_2"] = ndtr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.ndtr_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.ndtr_2'.")


check_valid('jax.scipy.special.ndtr', generated_inputs['jax.scipy.special.ndtr_2'], lib="jax", suffix=2)
