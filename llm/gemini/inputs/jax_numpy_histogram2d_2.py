
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram2d_inputs():
    list_of_inputs = []

    # Input 1
    x = np.random.uniform(0, 10, size=100).astype(np.float32)
    y = np.random.uniform(0, 10, size=100).astype(np.float32)
    bins = np.linspace(0, 10, 11).astype(np.float32)
    r = [[0.0, 10.0], [0.0, 10.0]]
    weights = np.random.uniform(0.5, 1.5, size=100).astype(np.float32)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 2
    x = np.random.randn(50).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    bins = np.array([-3.0, -1.0, 0.0, 1.0, 3.0]).astype(np.float32)
    r = [[-3.0, 3.0], [-3.0, 3.0]]
    weights = np.ones(50, dtype=np.float32)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 3: float64
    x = np.random.uniform(-5, 5, size=200).astype(np.float64)
    y = np.random.uniform(-5, 5, size=200).astype(np.float64)
    bins = np.arange(-5, 6, dtype=np.float64)
    r = [[-5.0, 5.0], [-5.0, 5.0]]
    weights = np.random.uniform(0.1, 1.0, size=200).astype(np.float64)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 4: integer x, y cast to float32
    x = np.random.randint(0, 20, size=80).astype(np.float32)
    y = np.random.randint(0, 20, size=80).astype(np.float32)
    bins = np.array([0, 5, 10, 15, 20], dtype=np.float32)
    r = [[0, 20], [0, 20]]
    weights = np.random.rand(80).astype(np.float32)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 5: larger size
    x = np.random.uniform(-100, 100, size=1000).astype(np.float32)
    y = np.random.uniform(-100, 100, size=1000).astype(np.float32)
    bins = np.linspace(-100, 100, 51).astype(np.float32)
    r = [[-100.0, 100.0], [-100.0, 100.0]]
    weights = np.random.uniform(1, 10, size=1000).astype(np.float32)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 6: negative values
    x = np.random.uniform(-10, -1, size=30).astype(np.float32)
    y = np.random.uniform(-10, -1, size=30).astype(np.float32)
    bins = np.array([-10.0, -8.0, -6.0, -4.0, -2.0, -1.0]).astype(np.float32)
    r = [[-10.0, -1.0], [-10.0, -1.0]]
    weights = np.random.exponential(1.0, size=30).astype(np.float32)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 7: small array size
    x = np.array([1.5, 2.5, 3.5]).astype(np.float32)
    y = np.array([4.5, 5.5, 6.5]).astype(np.float32)
    bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32)
    r = [[1.0, 7.0], [1.0, 7.0]]
    weights = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 8: float64 variables
    x = np.random.uniform(0, 1, size=40).astype(np.float64)
    y = np.random.uniform(0, 1, size=40).astype(np.float64)
    bins = np.linspace(0, 1, 6).astype(np.float64)
    r = [[0.0, 1.0], [0.0, 1.0]]
    weights = np.ones(40, dtype=np.float64)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 9: 2D bins (different bins for x and y)
    x = np.random.uniform(10, 20, size=150).astype(np.float32)
    y = np.random.uniform(100, 200, size=150).astype(np.float32)
    bins = np.array([[10., 12., 15., 18., 20.], [100., 120., 150., 180., 200.]]).astype(np.float32)
    r = [[10.0, 20.0], [100.0, 200.0]]
    weights = np.random.uniform(0.5, 1.5, size=150).astype(np.float32)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    # Input 10: 2D bins with float64
    x = np.random.uniform(-5, 5, size=60).astype(np.float64)
    y = np.random.uniform(0, 10, size=60).astype(np.float64)
    bins = np.array([[-5.0, -2.5, 0.0, 2.5, 5.0], [0.0, 2.5, 5.0, 7.5, 10.0]]).astype(np.float64)
    r = [[-5.0, 5.0], [0.0, 10.0]]
    weights = np.random.uniform(1, 2, size=60).astype(np.float64)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": r, "weights": weights, "density": density
    })

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_2"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_2'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_2'], lib="jax", suffix=2)
