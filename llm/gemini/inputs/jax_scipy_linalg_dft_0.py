
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dft_inputs():
    list_of_inputs = []

    # Input 1: size 2, sqrtn scaling, complex64
    input_dict = {
        'n': 2,
        'scale': 'sqrtn',
        'dtype': np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: size 4, n scaling, complex128
    input_dict = {
        'n': 4,
        'scale': 'n',
        'dtype': np.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: size 8, sqrtn scaling, complex128
    input_dict = {
        'n': 8,
        'scale': 'sqrtn',
        'dtype': np.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: size 16, n scaling, complex64
    input_dict = {
        'n': 16,
        'scale': 'n',
        'dtype': np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: size 1, sqrtn scaling, complex64
    input_dict = {
        'n': 1,
        'scale': 'sqrtn',
        'dtype': np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: size 3, n scaling, complex128
    input_dict = {
        'n': 3,
        'scale': 'n',
        'dtype': np.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: size 5, sqrtn scaling, complex128
    input_dict = {
        'n': 5,
        'scale': 'sqrtn',
        'dtype': np.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: size 10, n scaling, complex64
    input_dict = {
        'n': 10,
        'scale': 'n',
        'dtype': np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: size 32, sqrtn scaling, complex64
    input_dict = {
        'n': 32,
        'scale': 'sqrtn',
        'dtype': np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: size 64, n scaling, complex128
    input_dict = {
        'n': 64,
        'scale': 'n',
        'dtype': np.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.dft"] = dft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.dft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.dft'.")


check_valid('jax.scipy.linalg.dft', generated_inputs['jax.scipy.linalg.dft'], lib="jax", suffix=0)
