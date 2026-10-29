
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def correlate_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, full mode, auto method, default precision
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    in2 = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "method": "auto",
            "precision": "default",
        }
    )

    # Input 2: 1D arrays, same mode, direct method, high precision
    in1 = np.array([2.5, -1.0, 0.0, 3.2, -4.5], dtype=np.float64)
    in2 = np.array([1.0, -1.0, 2.0], dtype=np.float64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "method": "direct",
            "precision": "high",
        }
    )

    # Input 3: 1D arrays, valid mode, fft method, highest precision
    in1 = np.array([0.5, 1.5, -2.0, 3.0, 1.0, -1.0], dtype=np.float32)
    in2 = np.array([1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "method": "fft",
            "precision": "highest",
        }
    )

    # Input 4: 2D arrays, full mode, direct method, default precision
    in1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    in2 = np.array([[-1.0, 0.0], [1.0, -2.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "method": "direct",
            "precision": "default",
        }
    )

    # Input 5: 2D arrays, same mode, fft method, high precision
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "method": "fft",
            "precision": "high",
        }
    )

    # Input 6: 2D arrays, valid mode, auto method, highest precision
    in1 = np.random.randn(6, 7).astype(np.float64)
    in2 = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "method": "auto",
            "precision": "highest",
        }
    )

    # Input 7: 3D arrays, full mode, direct method, default precision
    in1 = np.random.randn(3, 3, 3).astype(np.float32)
    in2 = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "method": "direct",
            "precision": "default",
        }
    )

    # Input 8: 3D arrays, valid mode, fft method, high precision
    in1 = np.random.randn(4, 5, 6).astype(np.float32)
    in2 = np.random.randn(2, 2, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "method": "fft",
            "precision": "high",
        }
    )

    # Input 9: 1D complex arrays, same mode, auto method, default precision
    in1 = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    in2 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "method": "auto",
            "precision": "default",
        }
    )

    # Input 10: 2D complex arrays, valid mode, fft method, highest precision
    in1 = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex128
    )
    in2 = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(
        np.complex128
    )
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "method": "fft",
            "precision": "highest",
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.signal.correlate_1"] = correlate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.correlate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.correlate_1'.")


check_valid('jax.scipy.signal.correlate', generated_inputs['jax.scipy.signal.correlate_1'], lib="jax", suffix=1)
