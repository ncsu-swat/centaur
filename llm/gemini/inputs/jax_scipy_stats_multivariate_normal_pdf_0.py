
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_multivariate_normal_pdf_inputs():
    list_of_inputs = []

    # Input 1: 1D distribution (k=1), 1 point
    x = np.array([1.0], dtype=np.float32)
    mean = np.array([0.0], dtype=np.float32)
    cov = np.array([[1.0]], dtype=np.float32)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D distribution (k=2), 1 point
    x = np.array([0.5, -0.5], dtype=np.float32)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    cov = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D distribution (k=2), multiple points
    x = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]], dtype=np.float32)
    mean = np.array([1.0, 1.0], dtype=np.float32)
    cov = np.array([[1.5, 0.2], [0.2, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D distribution (k=3), 1 point, float64
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    mean = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cov = np.eye(3, dtype=np.float64)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 5D distribution (k=5), multiple batch dimensions
    x = np.random.randn(2, 4, 5).astype(np.float32)
    mean = np.zeros(5, dtype=np.float32)
    cov = np.eye(5, dtype=np.float32)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D distribution with diagonal covariance as 2D array
    x = np.array([1.0, -1.0], dtype=np.float32)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    cov = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D distribution with isotropic covariance matrix
    x = np.array([0.5, 0.5], dtype=np.float32)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    cov = np.array([[1.5, 0.0], [0.0, 1.5]], dtype=np.float32)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D distribution, batched points
    x = np.random.randn(10, 3).astype(np.float32)
    mean = np.array([0.1, -0.2, 0.5], dtype=np.float32)
    cov = np.array([[1.0, 0.1, 0.2], [0.1, 1.2, -0.1], [0.2, -0.1, 0.8]], dtype=np.float32)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D distribution (k=4), float64, diagonal covariance matrix
    x = np.random.randn(5, 4).astype(np.float64)
    mean = np.ones(4, dtype=np.float64)
    cov = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float64)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D distribution, negative mean, positive-definite covariance
    x = np.array([-2.0, -3.0], dtype=np.float32)
    mean = np.array([-1.0, -2.0], dtype=np.float32)
    cov = np.array([[3.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "mean": mean, "cov": cov}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.multivariate_normal.pdf"] = generate_multivariate_normal_pdf_inputs()

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
