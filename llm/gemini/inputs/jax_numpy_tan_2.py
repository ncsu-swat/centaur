
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tan_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive angle (pi/4)
    input_dict = {"x": float(np.pi / 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative angle (-pi/4)
    input_dict = {"x": float(-np.pi / 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Angle pi/6
    input_dict = {"x": float(np.pi / 6)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Angle pi/3
    input_dict = {"x": float(np.pi / 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative angle (-pi/3)
    input_dict = {"x": float(-np.pi / 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float value 1.0
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative arbitrary float
    input_dict = {"x": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small float near zero
    input_dict = {"x": 1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Angle pi
    input_dict = {"x": float(np.pi)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Large float value
    input_dict = {"x": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.tan_2"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tan_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tan_2'.")


check_valid('jax.numpy.tan', generated_inputs['jax.numpy.tan_2'], lib="jax", suffix=2)
