
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erfc_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32 (representing True)
    input_dict = {"x": np.float32(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar float32 (representing False)
    input_dict = {"x": np.float32(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D float64 array
    input_dict = {"x": np.array(1.0, dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float32 array
    input_dict = {"x": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 array
    input_dict = {"x": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array
    input_dict = {"x": np.random.choice([0.0, 1.0], size=(2, 3, 4)).astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float64 array
    input_dict = {"x": np.ones((2, 2, 2, 2), dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D float32 array
    input_dict = {"x": np.zeros((2, 1, 3, 1, 2), dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D float64 array
    input_dict = {"x": np.random.choice([0.0, 1.0], size=(100,)).astype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 identity-like array
    input_dict = {"x": np.eye(5, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float32 array
    input_dict = {"x": np.array([[[1.0], [0.0], [1.0], [0.0], [1.0]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.erfc_4"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erfc_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erfc_4'.")


check_valid('jax.lax.erfc', generated_inputs['jax.lax.erfc_4'], lib="jax", suffix=4)
