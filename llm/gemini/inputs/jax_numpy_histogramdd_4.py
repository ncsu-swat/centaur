
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogramdd_inputs():
    list_of_inputs = []

    # Input 1: 1D sample, float32, density=False
    sample = np.random.randn(50, 1).astype(np.float32)
    weights = np.random.uniform(0.5, 1.5, size=(50,)).astype(np.float32)
    input_dict = {
        "sample": sample,
        "bins": (10,),
        "range": ((-3.0, 3.0),),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D sample, float64, density=True
    sample = np.random.uniform(0.0, 10.0, size=(100, 2)).astype(np.float64)
    weights = np.ones((100,), dtype=np.float64)
    input_dict = {
        "sample": sample,
        "bins": (5, 5),
        "range": ((0.0, 10.0), (0.0, 10.0)),
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D sample, float32, negative and positive ranges
    sample = np.random.randn(120, 3).astype(np.float32)
    weights = np.random.rand(120).astype(np.float32)
    input_dict = {
        "sample": sample,
        "bins": (4, 4, 4),
        "range": ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D sample, small bin count per dimension
    sample = np.random.uniform(-1.0, 1.0, size=(80, 4)).astype(np.float32)
    weights = np.random.uniform(0.1, 2.0, size=(80,)).astype(np.float32)
    input_dict = {
        "sample": sample,
        "bins": (2, 3, 2, 3),
        "range": ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D sample, asymmetric bins and asymmetric ranges
    sample = np.random.randn(64, 2).astype(np.float32)
    weights = np.ones((64,), dtype=np.float32) * 0.5
    input_dict = {
        "sample": sample,
        "bins": (8, 4),
        "range": ((-4.0, 4.0), (-1.0, 3.0)),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D sample, large number of samples, density=True
    sample = np.random.randn(500, 1).astype(np.float64)
    weights = np.random.exponential(1.0, size=(500,)).astype(np.float64)
    input_dict = {
        "sample": sample,
        "bins": (20,),
        "range": ((-5.0, 5.0),),
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D sample with non-uniform ranges
    sample = np.random.uniform(0.0, 5.0, size=(200, 3)).astype(np.float32)
    weights = np.random.uniform(1.0, 2.0, size=(200,)).astype(np.float32)
    input_dict = {
        "sample": sample,
        "bins": (3, 5, 7),
        "range": ((0.0, 5.0), (1.0, 4.0), (2.0, 5.0)),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D sample with integer-valued sample converted to float
    sample = np.random.randint(-10, 10, size=(40, 2)).astype(np.float32)
    weights = np.ones((40,), dtype=np.float32)
    input_dict = {
        "sample": sample,
        "bins": (6, 6),
        "range": ((-10.0, 10.0), (-10.0, 10.0)),
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D sample, minimal number of bins
    sample = np.random.randn(30, 5).astype(np.float32)
    weights = np.random.rand(30).astype(np.float32)
    input_dict = {
        "sample": sample,
        "bins": (2, 2, 2, 2, 2),
        "range": (
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
        ),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D sample, float64, zero weights included
    sample = np.random.uniform(-5.0, 5.0, size=(75, 2)).astype(np.float64)
    weights = np.zeros((75,), dtype=np.float64)
    weights[:40] = 1.0
    input_dict = {
        "sample": sample,
        "bins": (10, 12),
        "range": ((-5.0, 5.0), (-5.0, 5.0)),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.histogramdd_4"] = histogramdd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogramdd_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogramdd_4'.")


check_valid('jax.numpy.histogramdd', generated_inputs['jax.numpy.histogramdd_4'], lib="jax", suffix=4)
