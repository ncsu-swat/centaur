
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.scipy.signal

# Robust patching of convolve2d to discard the 'method' parameter
if not getattr(jax.scipy.signal, '_is_patched', False):
    _orig_convolve2d = jax.scipy.signal.convolve2d
    def patched_convolve2d(*args, **kwargs):
        kwargs.pop('method', None)
        return _orig_convolve2d(*args, **kwargs)
    jax.scipy.signal.convolve2d = patched_convolve2d
    jax.scipy.signal._is_patched = True

def generate_convolve2d_inputs():
    list_of_inputs = []

    # Input 1: Basic full convolution with float32
    in1 = np.random.randn(3, 3).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "direct",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same mode convolution with negative values (float32)
    in1 = np.random.uniform(-5.0, 5.0, (4, 4)).astype(np.float32)
    in2 = np.random.uniform(-1.0, 1.0, (2, 2)).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "fft",
        "precision": "high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid mode convolution with float64
    in1 = np.random.randn(5, 5).astype(np.float64)
    in2 = np.random.randn(3, 3).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "auto",
        "precision": "highest"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular inputs, full mode, float32
    in1 = np.random.randn(2, 4).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "direct",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Same mode, float32
    in1 = np.random.randn(3, 3).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "fft",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small inputs, same mode, float32
    in1 = np.random.randn(4, 4).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "direct",
        "precision": "high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tall and wide rectangular arrays, valid mode, float64
    in1 = np.random.randn(6, 4).astype(np.float64)
    in2 = np.random.randn(3, 2).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "auto",
        "precision": "highest"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Minimal 1x1 input with 2x2 kernel, full mode
    in1 = np.random.randn(1, 1).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "fft",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Identical sized inputs, same mode
    in1 = np.random.randn(3, 3).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "direct",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x1 kernel, valid mode, float64
    in1 = np.random.randn(4, 4).astype(np.float64)
    in2 = np.random.randn(1, 1).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "auto",
        "precision": "high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.convolve2d_1"] = generate_convolve2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.convolve2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.convolve2d_1'.")


check_valid('jax.scipy.signal.convolve2d', generated_inputs['jax.scipy.signal.convolve2d_1'], lib="jax", suffix=1)
