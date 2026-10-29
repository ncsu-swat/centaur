
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class _PrecisionElement:
    def __init__(self, val="default"):
        self.val = val

    def __repr__(self):
        return self.val

    def __str__(self):
        return self.val

    def __array__(self, dtype=None, copy=None):
        return np.array(0, dtype=np.int32)

    def lower(self):
        return self.val.lower()

    def startswith(self, prefix):
        return self.val.startswith(prefix)

    def split(self, *args, **kwargs):
        return self.val.split(*args, **kwargs)

    def __eq__(self, other):
        return str(self.val) == str(other)

    def __hash__(self):
        return hash(self.val)


def correlate_inputs():
    list_of_inputs = []

    p_default = _PrecisionElement("default")
    p_high = _PrecisionElement("high")
    p_highest = _PrecisionElement("highest")

    # Input 1: 1D arrays, full mode, auto method
    in1 = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    in2 = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": (p_default, p_default),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, same mode, direct method
    in1 = np.array([-2.5, 0.0, 1.5, 3.2, -1.1], dtype=np.float32)
    in2 = np.array([1.0, -2.0], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": (p_high, p_high),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays, valid mode, fft method
    in1 = np.arange(10, dtype=np.float32) - 5.0
    in2 = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "fft",
        "precision": (p_highest, p_highest),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, full mode, direct method
    in1 = np.array([[1.0, 2.0], [-1.0, 0.0]], dtype=np.float32)
    in2 = np.array([[0.5, -0.5], [1.0, 1.5]], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "direct",
        "precision": (p_default, p_default),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays, same mode, fft method
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "fft",
        "precision": (p_default, p_high),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays, valid mode, auto method
    in1 = np.random.randn(6, 7).astype(np.float32)
    in2 = np.random.randn(2, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "auto",
        "precision": (p_high, p_default),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays, full mode, fft method
    in1 = np.random.randn(3, 4, 2).astype(np.float32)
    in2 = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "fft",
        "precision": (p_default, p_default),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays, same mode, direct method
    in1 = np.random.randn(4, 4, 3).astype(np.float32)
    in2 = np.random.randn(2, 2, 1).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": (p_highest, p_default),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays, valid mode, direct method
    in1 = np.random.randn(5, 5, 5).astype(np.float32)
    in2 = np.random.randn(3, 2, 4).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "direct",
        "precision": (p_high, p_highest),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D complex arrays, full mode, auto method
    in1 = np.array([1.0 + 2.0j, -2.0 + 0.5j, 3.0 - 1.0j], dtype=np.complex64)
    in2 = np.array([0.0 + 1.0j, 2.0 - 2.0j], dtype=np.complex64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": (p_default, p_default),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.correlate_2"] = correlate_inputs()

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
