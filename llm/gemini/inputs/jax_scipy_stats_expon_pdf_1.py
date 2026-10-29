
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expon_pdf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([0.5, 1.0, 2.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 2: 0D scalar tensors
    x = np.array(2.0, dtype=np.float32)
    loc = np.array(0.5, dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 3: 2D float64 arrays
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    loc = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64)
    scale = np.array([[1.0, 2.0], [0.5, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 4: Broadcasting shapes (2, 3) with (1,) and (2, 1)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([[1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 5: Values where x < loc (evaluation should yield 0)
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 6: 3D float32 tensors
    x = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 7: x exactly at loc
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    loc = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    scale = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 8: Non-trivial broadcasting across 3 dimensions
    x = np.linspace(0, 5, 12, dtype=np.float32).reshape(3, 1, 4)
    loc = np.zeros((1, 2, 1), dtype=np.float32)
    scale = np.ones((3, 2, 4), dtype=np.float32) * 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 9: Large and small scale values
    x = np.array([10.0, 100.0, 1000.0], dtype=np.float64)
    loc = np.array([0.0, 10.0, 100.0], dtype=np.float64)
    scale = np.array([0.01, 10.0, 500.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 10: 4D tensor with varying values
    x = np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(0.0, 0.5, size=(2, 2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.expon.pdf_1"] = expon_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.pdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.pdf_1'.")


check_valid('jax.scipy.stats.expon.pdf', generated_inputs['jax.scipy.stats.expon.pdf_1'], lib="jax", suffix=1)
