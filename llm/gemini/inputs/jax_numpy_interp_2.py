
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def interp_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D interpolation with float32
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    xp = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    fp = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(10.0, dtype=np.float32)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D target x, negative values, float64
    x = np.array([[-1.0, 0.5], [1.5, 3.5]], dtype=np.float64)
    xp = np.array([-2.0, 0.0, 2.0, 4.0], dtype=np.float64)
    fp = np.array([4.0, 0.0, 4.0, 16.0], dtype=np.float64)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(5.0, dtype=np.float64)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D target x, sine function interpolation, float32
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    xp = np.linspace(-5.0, 5.0, 10).astype(np.float32)
    fp = np.sin(xp)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(2 * np.pi, dtype=np.float32)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Target x has integer type, xp/fp are float32
    x = np.array([1, 2, 5], dtype=np.int32)
    xp = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    fp = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(4.0, dtype=np.float32)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large 1D array, exponential function
    x = np.linspace(-10.0, 10.0, 50).astype(np.float32)
    xp = np.linspace(-5.0, 5.0, 20).astype(np.float32)
    fp = np.exp(xp / 5.0).astype(np.float32)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(100.0, dtype=np.float32)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0-dimensional x (scalar array)
    x = np.array(1.5, dtype=np.float32)
    xp = np.array([1.0, 2.0], dtype=np.float32)
    fp = np.array([10.0, 20.0], dtype=np.float32)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(5.0, dtype=np.float32)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale float64 values
    x = np.array([1e5, -1e5], dtype=np.float64)
    xp = np.array([-1e6, 0.0, 1e6], dtype=np.float64)
    fp = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(1e7, dtype=np.float64)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D target x, squared function values
    x = np.random.uniform(0.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    xp = np.arange(11, dtype=np.float32)
    fp = np.arange(11, dtype=np.float32) ** 2
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(10.0, dtype=np.float32)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extreme target values with a large period
    x = np.array([-1000.0, 1000.0], dtype=np.float32)
    xp = np.array([-1.0, 1.0], dtype=np.float32)
    fp = np.array([-1.0, 1.0], dtype=np.float32)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(10000.0, dtype=np.float32)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16 data type
    x = np.array([0.25, 0.75], dtype=np.float16)
    xp = np.array([0.0, 1.0], dtype=np.float16)
    fp = np.array([0.0, 1.0], dtype=np.float16)
    left = "extrapolate"
    right = "extrapolate"
    period = np.array(2.0, dtype=np.float16)
    input_dict = {"x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.interp_2"] = interp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.interp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.interp_2'.")


check_valid('jax.numpy.interp', generated_inputs['jax.numpy.interp_2'], lib="jax", suffix=2)
