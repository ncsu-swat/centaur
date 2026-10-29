
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp
import numpy as np


def vectorize_inputs():
    list_of_inputs = []

    # Input 1: Scalar to scalar with 1D float array
    input_dict = {
        "pyfunc": (lambda x: x + 1.0,),
        "excluded": (),
        "signature": "()->()",
        "args": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar to scalar with 2D int array
    input_dict = {
        "pyfunc": (lambda x: x * 2,),
        "excluded": (),
        "signature": "()->()",
        "args": np.array([[-1, 2], [3, -4]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D reduction to scalar
    input_dict = {
        "pyfunc": (lambda x: jnp.sum(x),),
        "excluded": (),
        "signature": "(n)->()",
        "args": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D vector to 1D vector (mean centering), float64
    input_dict = {
        "pyfunc": (lambda x: x - jnp.mean(x),),
        "excluded": (),
        "signature": "(n)->(n)",
        "args": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D matrix reduction to scalar across 3D batch
    input_dict = {
        "pyfunc": (lambda x: jnp.max(x),),
        "excluded": (),
        "signature": "(m,n)->()",
        "args": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D prefix sum (cumulative sum) with int32
    input_dict = {
        "pyfunc": (lambda x: jnp.cumsum(x),),
        "excluded": (),
        "signature": "(n)->(n)",
        "args": np.array([1, 2, 3, 4, 5], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar absolute value with negative integers
    input_dict = {
        "pyfunc": (lambda x: jnp.abs(x),),
        "excluded": (),
        "signature": "()->()",
        "args": np.array([-5, -3, 0, 3, 5], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix transpose across batch
    input_dict = {
        "pyfunc": (lambda x: jnp.transpose(x),),
        "excluded": (),
        "signature": "(m,n)->(n,m)",
        "args": np.random.randn(2, 3, 4).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D product reduction
    input_dict = {
        "pyfunc": (lambda x: jnp.prod(x),),
        "excluded": (),
        "signature": "(n)->()",
        "args": np.array(
            [[1.0, 2.0], [-3.0, 4.0], [0.5, 2.0]], dtype=np.float32
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D vector to fixed-size output (min, max)
    input_dict = {
        "pyfunc": (lambda x: jnp.array([jnp.min(x), jnp.max(x)]),),
        "excluded": (),
        "signature": "(n)->(2)",
        "args": np.array(
            [[1.0, 5.0, 2.0], [-2.0, 3.0, 0.0]], dtype=np.float32
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D matrix column sum reduction
    input_dict = {
        "pyfunc": (lambda x: jnp.sum(x, axis=0),),
        "excluded": (),
        "signature": "(n,m)->(m)",
        "args": np.ones((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.vectorize_1"] = vectorize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vectorize_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vectorize_1'.")


check_valid('jax.numpy.vectorize', generated_inputs['jax.numpy.vectorize_1'], lib="jax", suffix=1)
