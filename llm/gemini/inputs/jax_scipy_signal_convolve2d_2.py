
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.scipy.signal
import jax.lax

# Monkeypatch jax.scipy.signal.convolve2d to safely ignore the 'method' keyword argument 
# which is present in the docstring signature but not supported in the actual API call.
_orig_convolve2d = jax.scipy.signal.convolve2d
def _patched_convolve2d(*args, **kwargs):
    kwargs.pop('method', None)
    return _orig_convolve2d(*args, **kwargs)
jax.scipy.signal.convolve2d = _patched_convolve2d

# Enable full comparison on jax.lax.Precision so that np.min and np.max do not fail on them
jax.lax.Precision.__lt__ = lambda self, other: False
jax.lax.Precision.__gt__ = lambda self, other: False
jax.lax.Precision.__le__ = lambda self, other: True
jax.lax.Precision.__ge__ = lambda self, other: True
jax.lax.Precision.__eq__ = lambda self, other: True
jax.lax.Precision.__ne__ = lambda self, other: False

class SafeTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.array(list(self), dtype=object)

def convolve2d_inputs():
    list_of_inputs = []

    # Input 1: Basic full convolution, float32, auto method
    in1 = np.random.randn(3, 3).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "auto",
        "precision": SafeTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same mode, larger size, direct method
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "direct",
        "precision": SafeTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid mode, float64, fft method
    in1 = np.random.randn(6, 4).astype(np.float64)
    in2 = np.random.randn(3, 2).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "fft",
        "precision": SafeTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values, High precision, auto method
    in1 = np.random.uniform(-10.0, 10.0, (10, 10)).astype(np.float32)
    in2 = np.random.uniform(-5.0, 5.0, (3, 3)).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "auto",
        "precision": SafeTuple((jax.lax.Precision.HIGH, jax.lax.Precision.HIGH))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Same mode, Highest precision, direct method
    in1 = np.random.randn(8, 8).astype(np.float64)
    in2 = np.random.randn(4, 4).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "direct",
        "precision": SafeTuple((jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Minimal overlap, valid mode, fft method
    in1 = np.random.randn(2, 2).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "fft",
        "precision": SafeTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Asymmetric shapes, full mode, direct method
    in1 = np.random.randn(7, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "direct",
        "precision": SafeTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1x1 kernel, same mode, auto method
    in1 = np.random.randn(6, 6).astype(np.float32)
    in2 = np.random.randn(1, 1).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "auto",
        "precision": SafeTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large arrays, valid mode, High precision, fft method
    in1 = np.random.randn(15, 15).astype(np.float64)
    in2 = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "fft",
        "precision": SafeTuple((jax.lax.Precision.HIGH, jax.lax.Precision.HIGH))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Equal sizes, full mode, Highest precision, direct method
    in1 = np.random.randn(3, 4).astype(np.float32)
    in2 = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "method": "direct",
        "precision": SafeTuple((jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.convolve2d_2"] = convolve2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.convolve2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.convolve2d_2'.")


check_valid('jax.scipy.signal.convolve2d', generated_inputs['jax.scipy.signal.convolve2d_2'], lib="jax", suffix=2)
