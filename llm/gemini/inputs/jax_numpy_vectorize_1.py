
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class CallableTuple(tuple):
    def __new__(cls, func):
        return super().__new__(cls, (func,))
    def __call__(self, *args, **kwargs):
        return self[0](*args, **kwargs)

def jax_numpy_vectorize_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D to 1D mapping (doubling)
    pyfunc = CallableTuple(lambda x: x * 2)
    excluded = ()
    signature = "(n)->(n)"
    args = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 2: Reduction (summing elements of core dimension)
    pyfunc = CallableTuple(lambda x: x.sum())
    excluded = ()
    signature = "(n)->()"
    args = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 3: Scalar element-wise mapping
    pyfunc = CallableTuple(lambda x: x + 1.0)
    excluded = ()
    signature = "()->()"
    args = np.random.randn(5).astype(np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 4: Reversing core dimension
    pyfunc = CallableTuple(lambda x: x[::-1])
    excluded = ()
    signature = "(n)->(n)"
    args = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 5: Trigonometric float64 mapping
    pyfunc = CallableTuple(lambda x: np.sin(x))
    excluded = ()
    signature = "()->()"
    args = np.random.randn(2, 2).astype(np.float64)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 6: Reduction (mean of core dimension)
    pyfunc = CallableTuple(lambda x: x.mean())
    excluded = ()
    signature = "(n)->()"
    args = np.random.randn(10).astype(np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 7: Scalar mapping with excluded first argument
    pyfunc = CallableTuple(lambda x: x * 0.5)
    excluded = (0,)
    signature = "()->()"
    args = np.array([10.0, 20.0], dtype=np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 8: Integer cubic mapping
    pyfunc = CallableTuple(lambda x: x**3)
    excluded = ()
    signature = "()->()"
    args = np.array([-1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 9: Core dimension max reduction with higher dimensional input
    pyfunc = CallableTuple(lambda x: x.max())
    excluded = ()
    signature = "(n)->()"
    args = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    # Input 10: Int64 shift mapping
    pyfunc = CallableTuple(lambda x: x + 5)
    excluded = ()
    signature = "()->()"
    args = np.array([[1, 2, 3]], dtype=np.int64)
    list_of_inputs.append({
        "pyfunc": pyfunc,
        "excluded": excluded,
        "signature": signature,
        "args": args
    })

    return list_of_inputs

generated_inputs["jax.numpy.vectorize_1"] = jax_numpy_vectorize_inputs()

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
