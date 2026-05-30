
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iterable_inputs():
    list_of_inputs = []

    # Input 1: Python True
    list_of_inputs.append({"y": True})

    # Input 2: Python False
    list_of_inputs.append({"y": False})

    # Input 3: numpy bool_ True
    list_of_inputs.append({"y": np.bool_(True)})

    # Input 4: numpy bool_ False
    list_of_inputs.append({"y": np.bool_(False)})

    # Input 5: 0D numpy array True
    list_of_inputs.append({"y": np.array(True, dtype=np.bool_)})

    # Input 6: 0D numpy array False
    list_of_inputs.append({"y": np.array(False, dtype=np.bool_)})

    # Input 7: 1D numpy array with single True
    list_of_inputs.append({"y": np.array([True], dtype=np.bool_)})

    # Input 8: 1D numpy array with True and False
    list_of_inputs.append({"y": np.array([True, False], dtype=np.bool_)})

    # Input 9: 2D numpy array with True
    list_of_inputs.append({"y": np.array([[True]], dtype=np.bool_)})

    # Input 10: 2D numpy array 2x2
    list_of_inputs.append({"y": np.array([[True, False], [False, True]], dtype=np.bool_)})

    return list_of_inputs

generated_inputs["jax.numpy.iterable_6"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_6'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_6'], lib="jax", suffix=6)
