import copy
import random
import unittest

import torch
import torch.nn as nn

from float8_utils import (
    float32_to_float8,
    float8_to_float32,
    E4M3,
    E5M2,
    compute_error,
    tensor_to_scale,
)
from float8_tensor import Float8Tensor
from float8_linear import Float8Linear


if __name__ == "__main__":
    # Create a float32 precision tensor
    x = torch.randn(128)

    # Create FP8 tensor from the above full precision tensor
    x_fp8 = Float8Tensor.from_float32(x, tensor_to_scale(x, E4M3), E4M3)

    # Inspect fp8 tensor before nn.uniform_
    print(x_fp8)

    # Call init.uniform_
    torch.nn.init.uniform_(x_fp8)

    # Inspect fp8 tensor after nn.uniform_
    print(x_fp8)