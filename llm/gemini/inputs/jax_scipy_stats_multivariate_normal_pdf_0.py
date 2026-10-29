
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multivariate_normal_pdf_inputs():
    list_of_inputs = []

    # Input 1: 1D case, standard normal
    x = np.array([0.5], dtype=np.float32)
    mean = np.array([0.0], dtype=np.float32)
    cov = np.array([[1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 2: 2D standard normal single point
    x = np.array([1.0, -1.0], dtype=np.float32)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    cov = np.eye(2, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 3: 2D normal with non-zero mean and general positive definite covariance
    x = np.array([-0.5, 1.2], dtype=np.float64)
    mean = np.array([1.0, -2.0], dtype=np.float64)
    cov = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 4: 3D distribution, batched x points
    x = np.random.randn(10, 3).astype(np.float32)
    mean = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    m = np.random.randn(3, 3).astype(np.float32)
    cov = m @ m.T + np.eye(3, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 5: 4D distribution, single point, float64
    x = np.array([0.1, -0.3, 0.5, -0.7], dtype=np.float64)
    mean = np.zeros(4, dtype=np.float64)
    m = np.random.randn(4, 4).astype(np.float64)
    cov = m @ m.T + np.eye(4, dtype=np.float64) * 0.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 6: 2D distribution with multidimensional batch of x (shape (4, 5, 2))
    x = np.random.randn(4, 5, 2).astype(np.float32)
    mean = np.array([0.5, -0.5], dtype=np.float32)
    cov = np.array([[1.5, 0.3], [0.3, 0.8]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 7: Batched evaluation where mean and cov are also batched
    batch_size = 5
    dim = 2
    x = np.random.randn(batch_size, dim).astype(np.float32)
    mean = np.random.randn(batch_size, dim).astype(np.float32)
    m = np.random.randn(batch_size, dim, dim).astype(np.float32)
    cov = np.einsum("bij,bkj->bik", m, m) + np.tile(
        np.eye(dim, dtype=np.float32), (batch_size, 1, 1)
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 8: Higher-dimensional (8D) single point
    dim = 8
    x = np.ones((dim,), dtype=np.float32)
    mean = np.full((dim,), -0.5, dtype=np.float32)
    m = np.random.randn(dim, dim).astype(np.float32)
    cov = m @ m.T + np.eye(dim, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 9: Diagonal covariance matrix with negative values in x and mean
    dim = 3
    x = np.array([-1.5, -2.5, -3.5], dtype=np.float32)
    mean = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    cov = np.diag(np.array([0.5, 1.5, 2.5], dtype=np.float32))
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    # Input 10: 2D distribution with batch of 1
    x = np.array([[0.0, 0.0]], dtype=np.float64)
    mean = np.array([0.0, 0.0], dtype=np.float64)
    cov = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "mean": mean, "cov": cov})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.multivariate_normal.pdf"] = multivariate_normal_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.multivariate_normal.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.multivariate_normal.pdf'.")


check_valid('jax.scipy.stats.multivariate_normal.pdf', generated_inputs['jax.scipy.stats.multivariate_normal.pdf'], lib="jax", suffix=0)
