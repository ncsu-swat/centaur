
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: Python True
    list_of_inputs.append({"x": True})

    # Input 2: Python False
    list_of_inputs.append({"x": False})

    # Input 3: numpy boolean scalar True
    list_of_inputs.append({"x": np.bool_(True)})

    # Input 4: numpy boolean scalar False
    list_of_inputs.append({"x": np.bool_(False)})

    # Input 5: 1D boolean array
    list_of_inputs.append({"x": np.array([True, False, True], dtype=bool)})

    # Input 6: 2D boolean array
    list_of_inputs.append({"x": np.array([[True, False], [False, True]], dtype=bool)})

    # Input 7: 3D boolean array
    list_of_inputs.append({"x": np.ones((2, 3, 4), dtype=bool)})

    # Input 8: 4D boolean array of zeros
    list_of_inputs.append({"x": np.zeros((1, 2, 2, 3), dtype=bool)})

    # Input 9: Empty 1D boolean array
    list_of_inputs.append({"x": np.array([], dtype=bool)})

    # Input 10: 5D boolean array
    list_of_inputs.append({"x": np.ones((1, 1, 2, 2, 2), dtype=bool)})

    return list_of_inputs

generated_inputs["jax.numpy.iscomplexobj_4"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_4'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_4'], lib="jax", suffix=4)
