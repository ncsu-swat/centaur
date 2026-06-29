
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_logistic_logpdf_inputs():
    list_of_inputs = []

    # 1. 1D float32 array, loc=0.0, scale=1.0
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": 0.0, "scale": 1.0})

    # 2. 2D float32 array, loc=1.5, scale=2.5
    x = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": 1.5, "scale": 2.5})

    # 3. 3D float64 array, loc=-0.5, scale=0.5
    x = np.random.randn(2, 2, 2).astype(np.float64)
    list_of_inputs.append({"x": x, "loc": -0.5, "scale": 0.5})

    # 4. 1D float32 array with wider range, scale=50.0
    x = np.linspace(-100, 100, 10).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": 0.0, "scale": 50.0})

    # 5. Scalar represented as a 0D numpy array, scale=1.5
    x = np.array(1.23, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": -1.0, "scale": 1.5})

    # 6. High dimensional array, loc=2.0, scale=0.8
    x = np.random.uniform(-5, 5, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": 2.0, "scale": 0.8})

    # 7. 1D float64 array, loc=0.0, scale=0.1
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": 0.0, "scale": 0.1})

    # 8. 2D array with positive values, loc=1.0, scale=1.0
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": 1.0, "scale": 1.0})

    # 9. 2D array with negative values, loc=-1.0, scale=2.0
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": -1.0, "scale": 2.0})

    # 10. Large 4D array, loc=0.0, scale=3.0
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": 0.0, "scale": 3.0})

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.logpdf_2"] = generate_logistic_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.logpdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.logpdf_2'.")


check_valid('jax.scipy.stats.logistic.logpdf', generated_inputs['jax.scipy.stats.logistic.logpdf_2'], lib="jax", suffix=2)
