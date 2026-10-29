
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_logsf_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard Cauchy
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 2: 2D array with float64, non-zero loc and scale
    x = np.array([[-5.0, -2.5], [0.0, 2.5]], dtype=np.float64)
    loc = 1.5
    scale = 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 3: 3D array, small scale
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = -1.0
    scale = 0.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 4: Scalar tensor (0D array)
    x = np.array(3.14, dtype=np.float32)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 5: Large positive values in x
    x = np.array([10.0, 100.0, 1000.0], dtype=np.float64)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 6: Large negative values in x
    x = np.array([-1000.0, -100.0, -10.0], dtype=np.float64)
    loc = 5.0
    scale = 3.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 7: 4D array
    x = np.ones((2, 2, 2, 2), dtype=np.float32) * 0.5
    loc = 0.5
    scale = 1.2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 8: Integer tensor
    x = np.array([-3, -1, 0, 2, 5], dtype=np.int32)
    loc = 0.0
    scale = 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 9: Large scale parameter
    x = np.linspace(-10.0, 10.0, 15, dtype=np.float32)
    loc = -2.5
    scale = 50.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 10: 2D array with extreme negative loc
    x = np.array([[-100.0, 0.0], [50.0, 200.0]], dtype=np.float64)
    loc = -50.0
    scale = 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.logsf_2"] = cauchy_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logsf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logsf_2'.")


check_valid('jax.scipy.stats.cauchy.logsf', generated_inputs['jax.scipy.stats.cauchy.logsf_2'], lib="jax", suffix=2)
