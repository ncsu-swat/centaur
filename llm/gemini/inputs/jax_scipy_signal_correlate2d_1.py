
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def correlate2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, mode='full', default precision
    in1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    in2 = np.array([[0.5, -1.0], [2.0, 0.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "default",
        }
    )

    # Input 2: Different shapes, mode='same', high precision
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "high",
        }
    )

    # Input 3: Valid mode with rectangular inputs, highest precision
    in1 = np.random.randn(6, 4).astype(np.float32)
    in2 = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "highest",
        }
    )

    # Input 4: Float64 inputs, mode='full'
    in1 = np.arange(9, dtype=np.float64).reshape((3, 3)) - 4.0
    in2 = np.ones((2, 2), dtype=np.float64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "high",
        }
    )

    # Input 5: 1xN and Nx1 vectors as 2D arrays, mode='same'
    in1 = np.random.randn(1, 8).astype(np.float32)
    in2 = np.random.randn(1, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "default",
        }
    )

    # Input 6: Column vectors as 2D arrays, mode='valid'
    in1 = np.random.randn(7, 1).astype(np.float32)
    in2 = np.random.randn(3, 1).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "highest",
        }
    )

    # Input 7: Complex numbers, mode='full'
    in1 = (
        np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    ).astype(np.complex64)
    in2 = (
        np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    ).astype(np.complex64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "default",
        }
    )

    # Input 8: Integer types promoted, mode='same'
    in1 = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32)
    in2 = np.array([[1, 0], [0, 1]], dtype=np.int32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "default",
        }
    )

    # Input 9: Single-element kernel, mode='valid'
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.array([[2.5]], dtype=np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "high",
        }
    )

    # Input 10: Larger inputs with uppercase precision string, mode='full'
    in1 = np.random.uniform(-10.0, 10.0, size=(12, 10)).astype(np.float32)
    in2 = np.random.uniform(-5.0, 5.0, size=(4, 5)).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "boundary": "fill",
            "fillvalue": 0.0,
            "precision": "DEFAULT",
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.signal.correlate2d_1"] = correlate2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.correlate2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.correlate2d_1'.")


check_valid('jax.scipy.signal.correlate2d', generated_inputs['jax.scipy.signal.correlate2d_1'], lib="jax", suffix=1)
