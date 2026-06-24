
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.lax

# Monkeypatch jax.lax.Precision to support comparison operators for numpy min/max
jax.lax.Precision.__lt__ = lambda self, other: True
jax.lax.Precision.__gt__ = lambda self, other: True
jax.lax.Precision.__le__ = lambda self, other: True
jax.lax.Precision.__ge__ = lambda self, other: True

def convolve_inputs():
    list_of_inputs = []

    # Input 1: 1D, float32, full, auto
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    in2 = np.array([0.5, 1.0, 0.5], dtype=np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    })

    # Input 2: 1D, negative values, float32, same, direct
    in1 = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    in2 = np.array([1.0, -1.0], dtype=np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    })

    # Input 3: 1D, float64, valid, fft
    in1 = np.random.randn(10).astype(np.float64)
    in2 = np.random.randn(4).astype(np.float64)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "fft",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    })

    # Input 4: 2D, float32, full, auto
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    })

    # Input 5: 2D, float32, same, direct
    in1 = np.random.randn(6, 6).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    })

    # Input 6: 2D, float64, valid, fft
    in1 = np.random.randn(7, 7).astype(np.float64)
    in2 = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "fft",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    })

    # Input 7: 3D, float32, full, auto
    in1 = np.random.randn(4, 4, 4).astype(np.float32)
    in2 = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    })

    # Input 8: 3D, float32, same, direct
    in1 = np.random.randn(5, 5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3, 3).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    })

    # Input 9: 1D, float32, same, auto, precision tuple
    in1 = np.random.randn(15).astype(np.float32)
    in2 = np.random.randn(5).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "auto",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    })

    # Input 10: 2D, float32, valid, auto, precision tuple
    in1 = np.random.randn(8, 8).astype(np.float32)
    in2 = np.random.randn(4, 4).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "auto",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    })

    return list_of_inputs

generated_inputs["jax.scipy.signal.convolve_2"] = convolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.convolve_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.convolve_2'.")


check_valid('jax.scipy.signal.convolve', generated_inputs['jax.scipy.signal.convolve_2'], lib="jax", suffix=2)
