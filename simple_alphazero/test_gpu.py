"""
Test script to verify GPU usage.

This script tests if the GPU is being used correctly by the neural network.
"""

import torch
import time
from model.neural_net import create_model, board_to_tensor
import chess
import config

def test_gpu_performance():
    """Test GPU performance vs CPU performance."""
    print("Testing GPU vs CPU performance...")

    # Create a larger batch for more realistic workload
    batch_size = 128
    boards = [chess.Board() for _ in range(batch_size)]
    input_tensors = torch.stack([board_to_tensor(board) for board in boards])

    # Create models
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda')
        cpu_device = torch.device('cpu')

        # Create models
        cuda_model = create_model(device=cuda_device)
        cpu_model = create_model(device=cpu_device)

        # Copy weights to ensure fair comparison
        cpu_model.load_state_dict(cuda_model.state_dict())

        # Prepare inputs
        cuda_input = input_tensors.to(cuda_device)
        cpu_input = input_tensors.to(cpu_device)

        # Warm up
        print("Warming up models...")
        for _ in range(5):
            with torch.no_grad():
                cuda_model(cuda_input)
                cpu_model(cpu_input)

        # Test GPU performance
        print("Testing GPU performance...")
        iterations = 100
        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                cuda_model(cuda_input)
        cuda_time = time.time() - start_time

        # Test CPU performance
        print("Testing CPU performance...")
        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                cpu_model(cpu_input)
        cpu_time = time.time() - start_time

        print(f"Batch size: {batch_size}")
        print(f"GPU time for {iterations} batch inferences: {cuda_time:.4f} seconds")
        print(f"CPU time for {iterations} batch inferences: {cpu_time:.4f} seconds")
        print(f"GPU is {cpu_time/cuda_time:.2f}x faster than CPU")
    else:
        print("CUDA is not available. Cannot test GPU performance.")

def test_mixed_precision():
    """Test mixed precision performance."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot test mixed precision.")
        return

    print("Testing mixed precision performance...")

    # Create a larger batch for more realistic workload
    batch_size = 256
    boards = [chess.Board() for _ in range(batch_size)]
    input_tensor = torch.stack([board_to_tensor(board) for board in boards])

    # Create model
    device = torch.device('cuda')
    model = create_model(device=device)

    # Move input to device once to avoid transfer overhead
    input_tensor = input_tensor.to(device)

    # Warm up
    print("Warming up model...")
    for _ in range(5):
        with torch.no_grad():
            model(input_tensor)
        with torch.amp.autocast(device_type='cuda'):
            with torch.no_grad():
                model(input_tensor)

    # Test FP32 performance
    print("Testing FP32 performance...")
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            model(input_tensor)
    fp32_time = time.time() - start_time

    # Test mixed precision performance
    print("Testing mixed precision performance...")
    start_time = time.time()
    for _ in range(iterations):
        with torch.amp.autocast(device_type='cuda'):
            with torch.no_grad():
                model(input_tensor)
    mixed_time = time.time() - start_time

    print(f"Batch size: {batch_size}")
    print(f"FP32 time for {iterations} batch inferences: {fp32_time:.4f} seconds")
    print(f"Mixed precision time for {iterations} batch inferences: {mixed_time:.4f} seconds")
    print(f"Mixed precision is {fp32_time/mixed_time:.2f}x faster than FP32")

def print_gpu_info():
    """Print GPU information."""
    print("GPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        print(f"RTX 4070 optimizations enabled: {config.RTX4070_OPTIMIZATIONS}")

if __name__ == "__main__":
    print_gpu_info()
    print("\n" + "="*50 + "\n")
    test_gpu_performance()
    print("\n" + "="*50 + "\n")
    test_mixed_precision()
