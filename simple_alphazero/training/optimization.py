"""
Neural network training optimization.

This module handles the training of the neural network using examples
generated from self-play games.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
import sys
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Add parent directory to path for importing config
sys.path.append('..')
import config


class ChessDataset(Dataset):
    """
    Dataset for training the neural network with self-play examples.
    """

    def __init__(self, examples):
        """
        Initialize the dataset with examples.

        Args:
            examples: List of (state, policy, value) tuples
        """
        self.states = []
        self.policies = []
        self.values = []

        # Process examples
        for state, policy, value in examples:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.states)

    def __getitem__(self, idx):
        """Get a single example from the dataset."""
        return (
            self.states[idx],
            self.policies[idx],
            torch.tensor([self.values[idx]], dtype=torch.float)
        )


class Trainer:
    """
    Neural network trainer for the AlphaZero-style chess model.

    This class handles:
    - Training the neural network with examples from self-play
    - Tracking training metrics
    - Saving model checkpoints
    - CUDA graph optimization for repeated operations
    """

    def __init__(self, model, optimizer, device='cuda', batch_size=256,
                 num_epochs=10, checkpoint_dir='checkpoints', use_mixed_precision=True):
        """
        Initialize the trainer.

        Args:
            model: Neural network model to train
            optimizer: Optimizer for training
            device: Device to train on ('cuda' or 'cpu')
            batch_size: Batch size for training
            num_epochs: Number of epochs to train for
            checkpoint_dir: Directory to save checkpoints
            use_mixed_precision: Whether to use mixed precision training (FP16)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'

        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None

        # CUDA graph settings
        self.use_cuda_graphs = (device.type == 'cuda' and
                               config.RTX4070_OPTIMIZATIONS.get('use_cuda_graphs', False))
        self.cuda_graph_batch_size = config.RTX4070_OPTIMIZATIONS.get('cuda_graph_batch_size', 256)
        self.warmup_iterations = config.RTX4070_OPTIMIZATIONS.get('warmup_iterations', 3)

        # Set CUDA memory allocation settings for better performance
        if device.type == 'cuda' and torch.cuda.is_available():
            # Enable CUDA memory caching to reduce allocation overhead
            # Note: set_per_process_memory_fraction is not available in all PyTorch versions
            # Using a try-except block to handle compatibility
            try:
                torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use 95% of available GPU memory
            except (AttributeError, ImportError):
                # Fall back to older API or skip if not available
                try:
                    torch.cuda.set_per_process_memory_fraction(0.95)
                except (AttributeError, ImportError):
                    print("Warning: Could not set CUDA memory fraction - feature not available in this PyTorch version")

            # Set CUDA stream priority for better scheduling if available
            try:
                torch.cuda.set_stream_priority(priority=0)  # High priority
            except (AttributeError, ImportError):
                print("Warning: Could not set CUDA stream priority - feature not available in this PyTorch version")

        # CUDA graph objects
        self.static_inputs = None
        self.static_labels = None
        self.static_values = None
        self.graph = None
        self.stream = None
        self.capture_stream = None
        self.static_loss = None
        self.static_policy_loss = None
        self.static_value_loss = None
        self.graph_captured = False

        # CUDA graph capture error mode
        self.capture_error_mode = config.RTX4070_OPTIMIZATIONS.get('capture_error_mode', 'thread_local')

        if self.use_cuda_graphs and torch.cuda.is_available():
            try:
                self.graph = torch.cuda.CUDAGraph()
                self.stream = torch.cuda.Stream()
                print("CUDA Graph optimization enabled")
                print(f"Using capture error mode: {self.capture_error_mode}")

                # Set environment variables to help with CUDA graph debugging
                import os
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

                # Enable device-side assertions if specified in config
                if config.RTX4070_OPTIMIZATIONS.get('enable_cuda_dsa', False):
                    os.environ['TORCH_USE_CUDA_DSA'] = '1'
                    print("CUDA device-side assertions enabled")
            except Exception as e:
                print(f"Failed to initialize CUDA graph: {str(e)}")
                print("CUDA Graph optimization disabled")
                self.use_cuda_graphs = False

        # Training metrics
        self.train_losses = []
        self.policy_losses = []
        self.value_losses = []

    def _initialize_cuda_graph(self, states, policies, values):
        """
        Initialize and capture a CUDA graph for the training forward and backward pass.

        Args:
            states: Tensor of states
            policies: Tensor of policies
            values: Tensor of values
        """
        try:
            # Create static tensors for CUDA graph with contiguous memory layout
            # Using contiguous tensors improves memory access patterns
            self.static_inputs = states.clone().contiguous()
            self.static_labels = policies.clone().contiguous()
            self.static_values = values.clone().contiguous()

            # Pre-allocate output tensors to avoid memory allocations during graph capture
            self.static_loss = torch.zeros(1, device=self.device)
            self.static_policy_loss = torch.zeros(1, device=self.device)
            self.static_value_loss = torch.zeros(1, device=self.device)

            # Ensure CUDA is synchronized before warmup
            torch.cuda.synchronize()

            # Check if we should skip the first backward pass
            skip_first_backward = config.RTX4070_OPTIMIZATIONS.get('skip_first_backward_pass', True)

            # Warmup iterations to stabilize GPU performance
            print(f"Running {self.warmup_iterations} warmup iterations before CUDA graph capture...")
            for i in range(self.warmup_iterations):
                # Forward pass with mixed precision if enabled
                if self.use_mixed_precision:
                    with autocast(device_type='cuda'):
                        policy_logits, value_preds = self.model(self.static_inputs)
                        policy_loss = F.cross_entropy(policy_logits, self.static_labels)
                        value_loss = F.mse_loss(value_preds, self.static_values)
                        loss = policy_loss + value_loss

                    # Skip the first backward pass if configured to do so
                    if not (skip_first_backward and i == 0):
                        # Backward pass and optimize with gradient scaling
                        self.optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    # Standard forward pass
                    policy_logits, value_preds = self.model(self.static_inputs)
                    policy_loss = F.cross_entropy(policy_logits, self.static_labels)
                    value_loss = F.mse_loss(value_preds, self.static_values)
                    loss = policy_loss + value_loss

                    # Skip the first backward pass if configured to do so
                    if not (skip_first_backward and i == 0):
                        # Backward pass and optimize
                        self.optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
                        loss.backward()
                        self.optimizer.step()

                # Print progress for long warmup iterations
                if (i + 1) % max(1, self.warmup_iterations // 3) == 0:
                    print(f"  Warmup iteration {i + 1}/{self.warmup_iterations} completed")

            # Prepare for graph capture
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            # Create a dedicated stream for graph capture
            # This helps avoid stream synchronization issues
            capture_stream = torch.cuda.Stream()

            # Capture the graph
            print("Capturing CUDA graph...")

            # Make sure all operations are completed before capture
            torch.cuda.synchronize()

            # Use the dedicated capture stream
            with torch.cuda.stream(capture_stream):
                # Make sure optimizer state is clean before capture
                self.optimizer.zero_grad(set_to_none=True)

                # Synchronize before graph capture
                capture_stream.synchronize()

                # Capture the graph with specified error mode
                with torch.cuda.graph(self.graph, stream=capture_stream, capture_error_mode=self.capture_error_mode):
                    try:
                        if self.use_mixed_precision:
                            with autocast(device_type='cuda'):
                                policy_logits, value_preds = self.model(self.static_inputs)
                                policy_loss = F.cross_entropy(policy_logits, self.static_labels)
                                value_loss = F.mse_loss(value_preds, self.static_values)
                                self.static_loss.copy_(policy_loss + value_loss)
                                self.static_policy_loss.copy_(policy_loss)
                                self.static_value_loss.copy_(value_loss)

                            # Backward pass with gradient scaling
                            # Avoid using optimizer.step() inside graph capture to prevent stream conflicts
                            scaled_loss = self.scaler.scale(self.static_loss)
                            scaled_loss.backward()
                        else:
                            # Standard forward pass
                            policy_logits, value_preds = self.model(self.static_inputs)
                            policy_loss = F.cross_entropy(policy_logits, self.static_labels)
                            value_loss = F.mse_loss(value_preds, self.static_values)
                            self.static_loss.copy_(policy_loss + value_loss)
                            self.static_policy_loss.copy_(policy_loss)
                            self.static_value_loss.copy_(value_loss)

                            # Backward pass only, no optimizer step
                            self.static_loss.backward()
                    except Exception as e:
                        # This exception will be caught by the outer try-except block
                        print(f"Error during CUDA graph capture operations: {str(e)}")
                        raise

            # Ensure synchronization after graph capture
            torch.cuda.synchronize()
            print("CUDA graph captured successfully")

            # Store the capture stream for later use
            self.capture_stream = capture_stream

            # Set flag to indicate successful graph capture
            self.graph_captured = True

        except Exception as e:
            print(f"CUDA graph capture failed with error: {str(e)}")
            print("Falling back to standard execution without CUDA graphs")

            # Reset graph state
            self.static_inputs = None
            self.static_labels = None
            self.static_values = None
            self.graph = None
            self.stream = None
            self.use_cuda_graphs = False

            # Re-raise for debugging if needed
            # raise

    def _run_with_cuda_graph(self, states, policies, values):
        """
        Run a training step using the captured CUDA graph.

        Args:
            states: Tensor of states
            policies: Tensor of policies
            values: Tensor of values

        Returns:
            loss: Loss value
            policy_loss: Policy loss value
            value_loss: Value loss value
        """
        # Check if graph capture was successful
        if not hasattr(self, 'graph_captured') or not self.graph_captured:
            # Fall back to standard execution if graph capture failed
            return self._run_without_cuda_graph(states, policies, values)

        try:
            # Ensure input tensors are contiguous for optimal memory access
            if not states.is_contiguous():
                states = states.contiguous()
            if not policies.is_contiguous():
                policies = policies.contiguous()
            if not values.is_contiguous():
                values = values.contiguous()

            # Use the capture stream for all operations
            with torch.cuda.stream(self.capture_stream):
                # Copy input data to static tensors
                self.static_inputs.copy_(states, non_blocking=True)
                self.static_labels.copy_(policies, non_blocking=True)
                self.static_values.copy_(values, non_blocking=True)

                # Ensure copies are complete before replaying the graph
                self.capture_stream.synchronize()

                # Replay the graph
                self.graph.replay()

                # Apply optimizer step after graph replay
                # This is done outside the graph to avoid stream conflicts
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Ensure graph execution is complete
                self.capture_stream.synchronize()

            # Return loss values (already on CPU due to .item() call)
            return (self.static_loss.item(),
                    self.static_policy_loss.item(),
                    self.static_value_loss.item())

        except Exception as e:
            print(f"Error during CUDA graph replay: {str(e)}")
            print("Falling back to standard execution")

            # Disable CUDA graphs for future iterations
            self.use_cuda_graphs = False

            # Fall back to standard execution
            return self._run_without_cuda_graph(states, policies, values)

    def _run_without_cuda_graph(self, states, policies, values):
        """
        Run a training step without using CUDA graphs (fallback method).

        Args:
            states: Tensor of states
            policies: Tensor of policies
            values: Tensor of values

        Returns:
            loss: Loss value
            policy_loss: Policy loss value
            value_loss: Value loss value
        """
        # Forward pass with mixed precision if enabled
        if self.use_mixed_precision:
            # Zero gradients more efficiently
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                # Forward pass
                policy_logits, value_preds = self.model(states)

                # Calculate losses
                policy_loss = F.cross_entropy(policy_logits, policies)
                value_loss = F.mse_loss(value_preds, values)

                # Combined loss
                loss = policy_loss + value_loss

            # Backward pass and optimize with gradient scaling
            self.scaler.scale(loss).backward()

            # Apply gradient clipping to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Zero gradients more efficiently
            self.optimizer.zero_grad(set_to_none=True)

            # Standard forward pass
            policy_logits, value_preds = self.model(states)

            # Calculate losses
            policy_loss = F.cross_entropy(policy_logits, policies)
            value_loss = F.mse_loss(value_preds, values)

            # Combined loss
            loss = policy_loss + value_loss

            # Backward pass
            loss.backward()

            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimize
            self.optimizer.step()

        # Get loss values (detach from computation graph to save memory)
        loss_val = loss.detach().item()
        policy_loss_val = policy_loss.detach().item()
        value_loss_val = value_loss.detach().item()

        return loss_val, policy_loss_val, value_loss_val

    def train(self, examples):
        """
        Train the model on a batch of examples.

        Args:
            examples: List of (state, policy, value) tuples

        Returns:
            avg_loss: Average loss over all batches
        """
        # Create dataset and dataloader with optimized settings
        dataset = ChessDataset(examples)

        # Configure DataLoader with optimized settings for GPU
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.RTX4070_OPTIMIZATIONS.get('num_workers', 4),
            pin_memory=config.RTX4070_OPTIMIZATIONS.get('pin_memory', True),
            persistent_workers=True if config.RTX4070_OPTIMIZATIONS.get('num_workers', 4) > 0 else False,
            prefetch_factor=2 if config.RTX4070_OPTIMIZATIONS.get('num_workers', 4) > 0 else None,
            drop_last=False
        )

        # Set model to training mode
        self.model.train()

        # Track metrics for this training session
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        batch_count = 0

        # Train for specified number of epochs
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_batches = 0

            # Process each batch
            for states, policies, values in dataloader:
                # Move data to device
                # Check if states is already a tensor or a list of tensors
                if isinstance(states, list):
                    states = torch.stack(states).to(self.device)
                else:
                    # If states is already a tensor, just move it to the device
                    states = states.to(self.device)
                policies = torch.tensor(np.array(policies), dtype=torch.float32).to(self.device)
                values = values.to(self.device)

                # Check if we can use CUDA graphs
                use_graph = (self.use_cuda_graphs and
                            states.size(0) == self.cuda_graph_batch_size and
                            self.device.type == 'cuda')

                # Initialize CUDA graph if needed
                if use_graph and self.graph is not None and not self.graph_captured:
                    try:
                        self._initialize_cuda_graph(states, policies, values)
                    except Exception as e:
                        print(f"CUDA graph initialization failed: {str(e)}")
                        print("Falling back to standard execution")
                        self.use_cuda_graphs = False
                        use_graph = False

                # Run with CUDA graph if possible
                if use_graph and self.graph_captured:
                    loss, policy_loss, value_loss = self._run_with_cuda_graph(states, policies, values)
                else:
                    # Ensure input tensors are contiguous for optimal memory access
                    if not states.is_contiguous():
                        states = states.contiguous()
                    if not policies.is_contiguous():
                        policies = policies.contiguous()
                    if not values.is_contiguous():
                        values = values.contiguous()

                    # Forward pass with mixed precision if enabled
                    if self.use_mixed_precision:
                        # Zero gradients more efficiently
                        self.optimizer.zero_grad(set_to_none=True)

                        with autocast(device_type='cuda'):
                            # Forward pass
                            policy_logits, value_preds = self.model(states)

                            # Calculate losses
                            policy_loss = F.cross_entropy(policy_logits, policies)
                            value_loss = F.mse_loss(value_preds, values)

                            # Combined loss
                            loss = policy_loss + value_loss

                        # Backward pass and optimize with gradient scaling
                        self.scaler.scale(loss).backward()

                        # Apply gradient clipping to prevent exploding gradients
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Zero gradients more efficiently
                        self.optimizer.zero_grad(set_to_none=True)

                        # Standard forward pass
                        policy_logits, value_preds = self.model(states)

                        # Calculate losses
                        policy_loss = F.cross_entropy(policy_logits, policies)
                        value_loss = F.mse_loss(value_preds, values)

                        # Combined loss
                        loss = policy_loss + value_loss

                        # Backward pass
                        loss.backward()

                        # Apply gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        # Optimize
                        self.optimizer.step()

                    # Get loss values (detach from computation graph to save memory)
                    loss = loss.detach().item()
                    policy_loss = policy_loss.detach().item()
                    value_loss = value_loss.detach().item()

                # Update metrics
                epoch_loss += loss
                epoch_policy_loss += policy_loss
                epoch_value_loss += value_loss
                epoch_batches += 1

            # Calculate average losses for this epoch
            avg_epoch_loss = epoch_loss / max(1, epoch_batches)
            avg_epoch_policy_loss = epoch_policy_loss / max(1, epoch_batches)
            avg_epoch_value_loss = epoch_value_loss / max(1, epoch_batches)

            # Update total metrics
            total_loss += avg_epoch_loss
            total_policy_loss += avg_epoch_policy_loss
            total_value_loss += avg_epoch_value_loss
            batch_count += 1

            # Log progress
            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Loss: {avg_epoch_loss:.4f}, "
                  f"Policy Loss: {avg_epoch_policy_loss:.4f}, "
                  f"Value Loss: {avg_epoch_value_loss:.4f}")

        # Calculate overall average losses
        avg_loss = total_loss / max(1, batch_count)
        avg_policy_loss = total_policy_loss / max(1, batch_count)
        avg_value_loss = total_value_loss / max(1, batch_count)

        # Store metrics
        self.train_losses.append(avg_loss)
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)

        return avg_loss

    def save_checkpoint(self, iteration, filename=None):
        """
        Save a model checkpoint.

        Args:
            iteration: Current training iteration
            filename: Optional specific filename to use

        Returns:
            path: Path to the saved checkpoint
        """
        import os

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Default filename based on iteration
        if filename is None:
            filename = f"model_iter_{iteration}.pt"

        # Full path to checkpoint
        path = os.path.join(self.checkpoint_dir, filename)

        # Reset CUDA graph before saving to avoid issues
        if self.use_cuda_graphs and self.graph is not None:
            # Reset static tensors and graph
            self.static_inputs = None
            self.static_labels = None
            self.static_values = None
            self.static_loss = None
            self.static_policy_loss = None
            self.static_value_loss = None
            self.graph_captured = False

            try:
                self.graph.reset()
            except Exception as e:
                print(f"Warning: Failed to reset CUDA graph: {str(e)}")
                self.graph = None

        # Save model state, optimizer state, scaler state, and training metrics
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'use_mixed_precision': self.use_mixed_precision,
            'use_cuda_graphs': self.use_cuda_graphs,
            'cuda_graph_batch_size': self.cuda_graph_batch_size,
            'warmup_iterations': self.warmup_iterations,
        }

        # Add scaler state if using mixed precision
        if self.use_mixed_precision and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)

        print(f"Checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path):
        """
        Load a model checkpoint.

        Args:
            path: Path to the checkpoint

        Returns:
            iteration: The iteration number of the loaded checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Load model and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load mixed precision settings and scaler state if available
        saved_mixed_precision = checkpoint.get('use_mixed_precision', False)
        if saved_mixed_precision and self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            if self.scaler is None:
                self.scaler = GradScaler('cuda')
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load CUDA graph settings if available
        saved_use_cuda_graphs = checkpoint.get('use_cuda_graphs', False)
        if saved_use_cuda_graphs and self.use_cuda_graphs:
            self.cuda_graph_batch_size = checkpoint.get('cuda_graph_batch_size', self.cuda_graph_batch_size)
            self.warmup_iterations = checkpoint.get('warmup_iterations', self.warmup_iterations)

            # Reset CUDA graph state to ensure clean state after loading
            self.static_inputs = None
            self.static_labels = None
            self.static_values = None
            self.static_loss = None
            self.static_policy_loss = None
            self.static_value_loss = None
            self.graph_captured = False

            # Reset the graph if it exists
            if self.graph is not None:
                try:
                    self.graph.reset()
                except Exception as e:
                    print(f"Warning: Failed to reset CUDA graph during loading: {str(e)}")
                    # Create a new graph
                    try:
                        self.graph = torch.cuda.CUDAGraph()
                        self.stream = torch.cuda.Stream()
                        self.capture_stream = None
                    except Exception as e2:
                        print(f"Failed to create new CUDA graph: {str(e2)}")
                        self.use_cuda_graphs = False

        # Load training metrics
        self.train_losses = checkpoint.get('train_losses', [])
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])

        # Get iteration number
        iteration = checkpoint.get('iteration', 0)

        print(f"Checkpoint loaded from {path} (iteration {iteration})")
        return iteration
