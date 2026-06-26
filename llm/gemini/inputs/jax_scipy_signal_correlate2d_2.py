
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class SafeTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.array(list(self), dtype=object)

def generate_correlate2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 correlation with "full" mode
    in1 = np.random.randn(3, 3).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('default', 'default'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: "same" mode with negative and positive values
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('high', 'high'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: "valid" mode using float64 data type
    in1 = np.random.randn(4, 4).astype(np.float64)
    in2 = np.random.randn(2, 2).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('highest', 'highest'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large dimensions with "full" mode
    in1 = np.random.randn(6, 6).astype(np.float32)
    in2 = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('default', 'default'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular filters on "same" mode
    in1 = np.random.randn(10, 10).astype(np.float32)
    in2 = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('high', 'high'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Same size arrays with "valid" mode
    in1 = np.random.randn(2, 2).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('highest', 'highest'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Highly rectangular input dimensions
    in1 = np.random.randn(7, 3).astype(np.float32)
    in2 = np.random.randn(3, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('default', 'default'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1x1 filter size with "same" mode
    in1 = np.random.randn(8, 8).astype(np.float32)
    in2 = np.random.randn(1, 1).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('high', 'high'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element arrays with "valid" mode
    in1 = np.random.randn(1, 1).astype(np.float32)
    in2 = np.random.randn(1, 1).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('highest', 'highest'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High-precision float64 arrays with "same" mode
    in1 = np.random.randn(12, 12).astype(np.float64)
    in2 = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": SafeTuple(('default', 'default'))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.correlate2d_2"] = generate_correlate2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.correlate2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.correlate2d_2'.")


check_valid('jax.scipy.signal.correlate2d', generated_inputs['jax.scipy.signal.correlate2d_2'], lib="jax", suffix=2)
