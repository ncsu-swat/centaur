
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def comb_inputs():
    list_of_inputs = []

    # Input 1: Basic standard combination (N=5, k=2)
    input_dict = {
        "N": int(5),
        "k": int(2),
        "exact": False,
        "repetition": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Combination with repetition (N=10, k=3)
    input_dict = {
        "N": int(10),
        "k": int(3),
        "exact": False,
        "repetition": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boundary case k = 0
    input_dict = {
        "N": int(8),
        "k": int(0),
        "exact": False,
        "repetition": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boundary case N = 0, k = 0
    input_dict = {
        "N": int(0),
        "k": int(0),
        "exact": False,
        "repetition": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boundary case k == N
    input_dict = {
        "N": int(6),
        "k": int(6),
        "exact": False,
        "repetition": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: k > N without repetition
    input_dict = {
        "N": int(4),
        "k": int(7),
        "exact": False,
        "repetition": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Using numpy int32 and numpy bool
    input_dict = {
        "N": np.int32(12),
        "k": np.int32(4),
        "exact": np.bool_(False),
        "repetition": np.bool_(False),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using numpy int64 and repetition
    input_dict = {
        "N": np.int64(20),
        "k": np.int64(5),
        "exact": False,
        "repetition": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large N with repetition
    input_dict = {
        "N": int(50),
        "k": int(2),
        "exact": False,
        "repetition": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: N = 1, k = 1
    input_dict = {
        "N": int(1),
        "k": int(1),
        "exact": False,
        "repetition": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.special.comb_2"] = comb_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.comb_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.comb_2'.")


check_valid('jax.scipy.special.comb', generated_inputs['jax.scipy.special.comb_2'], lib="jax", suffix=2)
