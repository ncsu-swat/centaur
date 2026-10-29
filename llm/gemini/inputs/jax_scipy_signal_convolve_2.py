
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def convolve_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, mode='full', method='fft'
    in1 = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    in2 = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "method": "fft",
            "precision": (0, 0),
        }
    )

    # Input 2: 1D float64, mode='same', method='fft'
    in1 = np.linspace(-3.0, 3.0, 15, dtype=np.float64)
    in2 = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "method": "fft",
            "precision": (0, 0),
        }
    )

    # Input 3: 1D float32, mode='valid', method='fft'
    in1 = np.random.randn(20).astype(np.float32)
    in2 = np.random.randn(5).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "method": "fft",
            "precision": (0, 1),
        }
    )

    # Input 4: 2D float32, mode='full', method='fft'
    in1 = np.random.randn(6, 6).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "method": "fft",
            "precision": (1, 1),
        }
    )

    # Input 5: 2D float64, mode='same', method='fft'
    in1 = np.random.randn(8, 8).astype(np.float64)
    in2 = np.random.randn(4, 4).astype(np.float64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "method": "fft",
            "precision": (0, 0),
        }
    )

    # Input 6: 2D float32, mode='valid', method='fft'
    in1 = np.random.randn(10, 8).astype(np.float32)
    in2 = np.random.randn(3, 2).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "method": "fft",
            "precision": (0, 2),
        }
    )

    # Input 7: 3D float32, mode='full', method='fft'
    in1 = np.random.randn(4, 5, 6).astype(np.float32)
    in2 = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "method": "fft",
            "precision": (0, 0),
        }
    )

    # Input 8: 3D float64, mode='same', method='fft'
    in1 = np.random.randn(5, 5, 5).astype(np.float64)
    in2 = np.random.randn(3, 3, 3).astype(np.float64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "method": "fft",
            "precision": (1, 0),
        }
    )

    # Input 9: 3D float32, mode='valid', method='fft'
    in1 = np.random.randn(7, 7, 7).astype(np.float32)
    in2 = np.random.randn(3, 3, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "method": "fft",
            "precision": (0, 0),
        }
    )

    # Input 10: 4D float32, mode='same', method='fft'
    in1 = np.random.randn(3, 4, 4, 3).astype(np.float32)
    in2 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "method": "fft",
            "precision": (0, 0),
        }
    )

    # Input 11: 1D complex64, mode='full', method='fft'
    in1 = (np.random.randn(12) + 1j * np.random.randn(12)).astype(np.complex64)
    in2 = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "method": "fft",
            "precision": (0, 0),
        }
    )

    # Input 12: 2D complex128, mode='same', method='fft'
    in1 = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(
        np.complex128
    )
    in2 = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "method": "fft",
            "precision": (0, 0),
        }
    )

    return [copy.deepcopy(inp) for inp in list_of_inputs]


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
