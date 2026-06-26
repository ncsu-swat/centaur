
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def multigammaln_inputs():
    list_of_inputs = []

    # Input 1: d=1, a is a 1D float32 array (a > 0)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = np.array(1, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    # Input 2: d=2, a is a 1D float32 array where values > 0.5
    a = np.array([0.6, 1.5, 2.3], dtype=np.float32)
    d = np.array(2, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    # Input 3: d=3, a is a 2D float64 array where values > 1.0
    a = np.array([[1.1, 2.5], [3.2, 4.8]], dtype=np.float64)
    d = np.array(3, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    # Input 4: d=4, a is a scalar float32 where value > 1.5
    a = np.array(2.0, dtype=np.float32)
    d = np.array(4, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    # Input 5: d=1, a has larger dimensions, float64
    a = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float64)
    d = np.array(1, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    # Input 6: d=5, a where values > 2.0
    a = np.random.uniform(2.1, 10.0, size=(3,)).astype(np.float32)
    d = np.array(5, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    # Input 7: d=2, a is a large 2D float32 array
    a = np.random.uniform(0.6, 5.0, size=(10, 10)).astype(np.float32)
    d = np.array(2, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    # Input 8: d=3, high-precision float64 array
    a = np.random.uniform(1.1, 100.0, size=(5, 5)).astype(np.float64)
    d = np.array(3, dtype=np.int64)
    list_of_inputs.append({"a": a, "d": d})

    # Input 9: d=10, a where values > 4.5
    a = np.random.uniform(4.6, 20.0, size=(2, 2)).astype(np.float32)
    d = np.array(10, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    # Input 10: d=2, high magnitude inputs
    a = np.array([100.0, 500.0, 1000.0], dtype=np.float64)
    d = np.array(2, dtype=np.int32)
    list_of_inputs.append({"a": a, "d": d})

    return list_of_inputs

generated_inputs["jax.scipy.special.multigammaln_2"] = multigammaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.multigammaln_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.multigammaln_2'.")


check_valid('jax.scipy.special.multigammaln', generated_inputs['jax.scipy.special.multigammaln_2'], lib="jax", suffix=2)
