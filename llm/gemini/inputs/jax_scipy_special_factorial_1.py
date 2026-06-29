
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_special_factorial_inputs():
    list_of_inputs = []

    # Input 1: 0D tensor (scalar array), float32
    n = np.array(5.0, dtype=np.float32)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor, float32
    n = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor, int32
    n = np.array([0, 1, 5, 10, 15], dtype=np.int32)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor, float64
    n = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor, int64
    n = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D tensor with negative floats (non-integers)
    n = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor, float32
    n = np.random.uniform(0.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D tensor, small positive values
    n = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D tensor, mix of zeros and integers
    n = np.array([0, 10, 20], dtype=np.int32)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D tensor, random float64
    n = np.random.uniform(1.0, 5.0, size=(3, 3)).astype(np.float64)
    input_dict = {"n": n, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.factorial_1"] = jax_scipy_special_factorial_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.factorial_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.factorial_1'.")


check_valid('jax.scipy.special.factorial', generated_inputs['jax.scipy.special.factorial_1'], lib="jax", suffix=1)
