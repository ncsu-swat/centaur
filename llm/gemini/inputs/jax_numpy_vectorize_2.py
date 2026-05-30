
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax.numpy as jnp
import copy

def vectorize_inputs():
    list_of_inputs = []

    # Input 1
    pyfunc = (lambda x: x + 1,)
    excluded = []
    signature = "()->()"
    args = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 2
    pyfunc = (lambda x: x * 2.0,)
    excluded = []
    signature = "()->()"
    args = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 3
    pyfunc = (lambda x: jnp.sum(x),)
    excluded = []
    signature = "(n)->()"
    args = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 4
    pyfunc = (lambda x: x[::-1],)
    excluded = []
    signature = "(n)->(n)"
    args = np.arange(12).reshape(3, 4).astype(np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 5
    pyfunc = (lambda x: jnp.mean(x),)
    excluded = []
    signature = "(n)->()"
    args = np.random.randn(5, 10).astype(np.float64)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 6
    pyfunc = (lambda x: x * x,)
    excluded = []
    signature = "()->()"
    args = np.array([-1, 0, 1], dtype=np.int64)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 7
    pyfunc = (lambda x: jnp.max(x),)
    excluded = []
    signature = "(n)->()"
    args = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 8
    pyfunc = (lambda x: x + 10,)
    excluded = []
    signature = "()->()"
    args = np.ones((2, 2, 2), dtype=np.int32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 9
    pyfunc = (lambda x: jnp.std(x),)
    excluded = []
    signature = "(n)->()"
    args = np.random.randn(4, 8).astype(np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 10
    pyfunc = (lambda x: x * 0.5,)
    excluded = []
    signature = "()->()"
    args = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    return list_of_inputs

generated_inputs["jax.numpy.vectorize_2"] = vectorize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vectorize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vectorize_2'.")


check_valid('jax.numpy.vectorize', generated_inputs['jax.numpy.vectorize_2'], lib="jax", suffix=2)
