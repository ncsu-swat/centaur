
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_le_inputs():
    list_of_inputs = []

    # Input 1: positive standard Python integers
    input_dict = {"x": 5, "y": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: reversed order positive standard Python integers
    input_dict = {"x": 10, "y": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative standard Python integers
    input_dict = {"x": -15, "y": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: equal standard Python integers
    input_dict = {"x": 0, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy int32 scalars
    input_dict = {"x": np.int32(100), "y": np.int32(200)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy int64 scalars, negative values
    input_dict = {"x": np.int64(-50), "y": np.int64(-100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy int32 scalars, equal values
    input_dict = {"x": np.int32(50), "y": np.int32(50)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: another set of standard Python integers
    input_dict = {"x": 127, "y": 127}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large standard Python integers
    input_dict = {"x": 123456789, "y": 987654321}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy int32 scalars with negative bounds
    input_dict = {"x": np.int32(-32000), "y": np.int32(32000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: mix of positive and negative, python ints
    input_dict = {"x": -42, "y": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.le_2"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_2'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_2'], lib="jax", suffix=2)
