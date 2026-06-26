
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Patch jax.lax.Precision to support comparison operators so that np.min/np.max
# can process them without raising TypeErrors.
jax.lax.Precision.__le__ = lambda self, other: self.name <= other.name if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__lt__ = lambda self, other: self.name < other.name if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__ge__ = lambda self, other: self.name >= other.name if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__gt__ = lambda self, other: self.name > other.name if isinstance(other, jax.lax.Precision) else NotImplemented

def jax_scipy_signal_correlate_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, mode='full', method='auto'
    in1 = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    in2 = np.array([1, 3, 2], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, mode='same', method='direct'
    in1 = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    in2 = np.array([0.5, -0.5], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays, mode='valid', method='fft'
    in1 = np.random.randn(10).astype(np.float32)
    in2 = np.random.randn(4).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "fft",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, mode='full', method='direct', float64 precision
    in1 = np.random.randn(5, 5).astype(np.float64)
    in2 = np.random.randn(3, 3).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "direct",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays, mode='same', method='fft'
    in1 = np.random.randn(6, 6).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "fft",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays, mode='valid', method='auto'
    in1 = np.random.randn(10, 8).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "auto",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays, mode='full', method='direct'
    in1 = np.random.randn(4, 4, 4).astype(np.float32)
    in2 = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "direct",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays, mode='same', method='fft'
    in1 = np.random.randn(5, 5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "fft",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex inputs, 1D
    in1 = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    in2 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex inputs, 2D
    in1 = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    in2 = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.correlate_2"] = jax_scipy_signal_correlate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.correlate_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.correlate_2'.")


check_valid('jax.scipy.signal.correlate', generated_inputs['jax.scipy.signal.correlate_2'], lib="jax", suffix=2)
