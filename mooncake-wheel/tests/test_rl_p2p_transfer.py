"""
Test suite for P2PTransferSender and P2PTransferReceiver with Qwen3-32B model.

This test validates:
1. Correctness test with small tensors
2. Performance test with Qwen3-32B model (1to1, 1to2, 2to1)

Test scenarios:
- Correctness: 128x128 tensor transfer
- 1to1: 1 sender + 1 receiver
- 1to2: 1 sender + 2 receivers
- 2to1: 2 senders + 1 receiver

Model: Qwen3-32B
- 64 transformer layers (8 layers for testing)
- Hidden size: 5120
- Intermediate size: 27648
- Vocab size: 152064
- Total parameters: ~32B
- Total memory (fp16): ~64GB
"""

import gc
import logging
import os
import time
import unittest
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.multiprocessing as mp

from mooncake.rl_p2p_transfer_manager import P2PTransferReceiver, P2PTransferSender

logger = logging.getLogger(__name__)

# Set multiprocessing start method
mp.set_start_method("spawn", force=True)


def create_simple_weight(size: int, device: torch.device, dtype=torch.float16) -> torch.Tensor:
    """Create a simple weight tensor for correctness testing."""
    return torch.randn(size, size, device=device, dtype=dtype)


@dataclass
class Qwen3_32BConfig:
    """Qwen3-32B model configuration."""
    num_layers: int = 64
    hidden_size: int = 5120
    intermediate_size: int = 27648
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    vocab_size: int = 152064

    def get_layer_param_counts(self) -> Dict[str, int]:
        """Get parameter counts for each layer type."""
        return {
            "qkv_proj": self.hidden_size * (self.hidden_size + 2 * (self.hidden_size // self.num_attention_heads) * self.num_key_value_heads),
            "o_proj": self.hidden_size * self.hidden_size,
            "gate_up_proj": self.hidden_size * self.intermediate_size * 2,
            "down_proj": self.intermediate_size * self.hidden_size,
            "ln": self.hidden_size * 2,
        }

    def get_total_params(self) -> int:
        """Calculate total parameter count."""
        layer_params = self.get_layer_param_counts()
        params_per_layer = sum(layer_params.values())
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.vocab_size * self.hidden_size
        total = embedding_params + (params_per_layer * self.num_layers) + lm_head_params
        return total

    def get_total_size_bytes(self, dtype=torch.float16) -> int:
        """Get total model size in bytes."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        return self.get_total_params() * element_size


def create_qwen3_layer_weights(config: Qwen3_32BConfig, device: torch.device, dtype=torch.float16) -> Dict[str, torch.Tensor]:
    """Create mock weights for a single Qwen3-32B transformer layer."""
    weights = {}
    qkv_size = config.hidden_size + 2 * (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads
    weights["qkv_proj"] = torch.randn(qkv_size, config.hidden_size, device=device, dtype=dtype)
    weights["o_proj"] = torch.randn(config.hidden_size, config.hidden_size, device=device, dtype=dtype)
    weights["gate_up_proj"] = torch.randn(config.intermediate_size * 2, config.hidden_size, device=device, dtype=dtype)
    weights["down_proj"] = torch.randn(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)
    weights["ln1"] = torch.randn(config.hidden_size, device=device, dtype=dtype)
    weights["ln2"] = torch.randn(config.hidden_size, device=device, dtype=dtype)
    return weights


def create_qwen3_full_model(config: Qwen3_32BConfig, device: torch.device, dtype=torch.float16) -> Dict[str, torch.Tensor]:
    """Create full Qwen3-32B model weights (using 8 layers for testing)."""
    weights = {}
    weights["embedding"] = torch.randn(config.vocab_size, config.hidden_size, device=device, dtype=dtype)

    num_test_layers = min(config.num_layers, 8)
    for layer_idx in range(num_test_layers):
        layer_weights = create_qwen3_layer_weights(config, device, dtype)
        for name, weight in layer_weights.items():
            weights[f"layer_{layer_idx}_{name}"] = weight

    weights["lm_head"] = torch.randn(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
    return weights


# ============================================================================
# Correctness Test with Small Tensors
# ============================================================================

def simple_sender_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    session_info_queue: mp.Queue,
    hostname: str = "127.0.0.1",
):
    """Simplified sender process for correctness testing."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        sender = P2PTransferSender(hostname=hostname, gpu_id=rank, pool_size=1)

        # Single small weight (128x128 fp16 = 32KB)
        weight = create_simple_weight(128, device, dtype=torch.float16)

        total_size_kb = weight.numel() * weight.element_size() / 1e3
        logger.info(f"[SimpleSender] Weight size: {total_size_kb:.2f}KB")

        # Register memory with sender
        ptr = weight.data_ptr()
        length = weight.numel() * weight.element_size()
        sender.batch_register_ptrs(
            names=["test_weight"],
            ptrs=[ptr],
            lens=[length]
        )

        # Share session info with receiver
        session_info_queue.put((rank, sender.metadata_session_id))
        logger.info(f"[SimpleSender] Sent metadata_session_id: {sender.metadata_session_id}")

        barrier.wait()

        # Wait for receiver to complete transfer
        barrier.wait()

        logger.info(f"[SimpleSender] Transfer completed successfully")

    except Exception as e:
        logger.error(f"[SimpleSender] Error: {e}", exc_info=True)
        result_queue.put(("sender_error", str(e)))


def simple_receiver_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    session_info_queue: mp.Queue,
):
    """Simplified receiver process for correctness testing."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[SimpleReceiver-{rank}] Using Receiver on GPU {rank}")

        receiver = P2PTransferReceiver(hostname="127.0.0.1", gpu_id=rank, pool_size=1)

        weight = create_simple_weight(128, device, dtype=torch.float16)
        original_data = weight.clone()

        # Wait for sender's session_id
        sender_rank, sender_session_id = session_info_queue.get(timeout=10)
        logger.info(f"[SimpleReceiver-{rank}] Got sender session_id from rank {sender_rank}: {sender_session_id}")

        barrier.wait()

        ptr = weight.data_ptr()
        length = weight.numel() * weight.element_size()

        handle = receiver.submit_transfer_task(
            name="test_weight",
            metadata_session_id=sender_session_id,
            ptr=ptr,
            length=length,
        )
        handle.wait()
        torch.cuda.synchronize()

        # Verify data changed
        data_changed = not torch.equal(weight, original_data)
        logger.info(f"[SimpleReceiver-{rank}] Data changed: {data_changed}")

        result_queue.put((f"receiver_{rank}_success", data_changed))

        barrier.wait()

    except Exception as e:
        logger.error(f"[SimpleReceiver-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"receiver_{rank}_error", str(e)))


def simple_worker_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    session_info_queue: mp.Queue,
):
    """Entry point for simple correctness test worker."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        simple_sender_process(rank, world_size, result_queue, barrier, session_info_queue, "127.0.0.1")
    else:
        simple_receiver_process(rank, world_size, result_queue, barrier, session_info_queue)


class TestP2PTransferCorrectness(unittest.TestCase):
    """Basic correctness tests with small weights."""

    def test_sender_receiver_correctness(self):
        """Test P2PTransferSender and P2PTransferReceiver correctness with single small weight."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        world_size = 2

        logger.info("\n" + "="*80)
        logger.info("Testing P2PTransferSender/Receiver Correctness (single 128x128 weight)")
        logger.info("="*80 + "\n")

        result_queue = mp.Queue()
        session_info_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            simple_worker_process,
            args=(world_size, result_queue, barrier, session_info_queue),
            nprocs=world_size,
            join=False,
        )

        results = {}
        timeout = 30
        start_time = time.time()

        while len(results) < (world_size - 1):
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(5, remaining_time))
                results[key] = value
            except Exception:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        self.assertIn("receiver_1_success", results)
        self.assertTrue(results["receiver_1_success"], "Data should have changed after transfer")

        logger.info("✓ P2PTransferSender/Receiver correctness test passed\n")

        result_queue.close()
        result_queue.join_thread()
        session_info_queue.close()
        session_info_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# Performance Tests with Qwen3-32B
# ============================================================================

def sender_process(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    session_info_queue: mp.Queue,
    hostname: str = "127.0.0.1",
    weight_slice: tuple = None,  # (start_idx, end_idx) or None for all weights
    num_receivers: int = 1,  # Number of receivers that need the session_id
):
    """Sender process for performance testing."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[Sender-{rank}] Initializing on GPU {rank}")

        sender = P2PTransferSender(hostname=hostname, gpu_id=rank)

        logger.info(f"[Sender-{rank}] Creating Qwen3-32B model weights...")
        all_weights = create_qwen3_full_model(config, device, dtype=torch.float16)

        # Select weight slice if specified (for multi-sender scenarios)
        if weight_slice is not None:
            start_idx, end_idx = weight_slice
            weight_items = list(all_weights.items())
            weights = dict(weight_items[start_idx:end_idx])
            logger.info(f"[Sender-{rank}] Using weight slice [{start_idx}:{end_idx}], {len(weights)} weights")
        else:
            weights = all_weights

        total_params = sum(w.numel() for w in weights.values())
        total_size_mb = sum(w.numel() * w.element_size() for w in weights.values()) / 1e6
        logger.info(f"[Sender-{rank}] Model created: {total_params/1e9:.2f}B params, {total_size_mb:.2f}MB")

        # Register all memory regions with sender
        names = []
        ptrs = []
        lens = []
        for weight_name, weight_tensor in weights.items():
            names.append(weight_name)
            ptrs.append(weight_tensor.data_ptr())
            lens.append(weight_tensor.numel() * weight_tensor.element_size())

        sender.batch_register_ptrs(names=names, ptrs=ptrs, lens=lens)
        logger.info(f"[Sender-{rank}] Registered {len(names)} memory regions")

        # Wait for all senders to be ready before sending session IDs
        # This ensures session IDs are sent in rank order
        barrier.wait()

        # Share metadata_session_id with ALL receivers
        # Each receiver needs to get the session_id, so put it num_receivers times
        for _ in range(num_receivers):
            session_info_queue.put((rank, sender.metadata_session_id))
        logger.info(f"[Sender-{rank}] Sent metadata_session_id to {num_receivers} receiver(s): {sender.metadata_session_id}")

        for update_idx in range(num_updates):
            update_start = time.time()

            # Sender just waits - receiver will initiate transfers
            barrier.wait()

            update_end = time.time()
            logger.info(
                f"[Sender-{rank}] Update {update_idx + 1}/{num_updates} completed in "
                f"{(update_end - update_start)*1000:.2f}ms"
            )

        logger.info(f"[Sender-{rank}] All updates completed")

    except Exception as e:
        logger.error(f"[Sender-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"sender_{rank}_error", str(e)))


def receiver_process(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    session_info_queue: mp.Queue,
    num_senders: int = 1,
    weight_slices: list = None,  # List of (start_idx, end_idx) for each sender
):
    """Receiver process for performance testing."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[Receiver-{rank}] Initializing on GPU {rank}")

        receiver = P2PTransferReceiver(hostname="127.0.0.1", gpu_id=rank)
        logger.info(f"[Receiver-{rank}] Using Receiver with {receiver.thread_pool_size} worker threads")

        # Create one complete weight buffer (for 1to1 or 1to2 scenarios)
        # or separate buffers per sender (for 2to1 scenarios with weight_slices)
        if weight_slices is None:
            # 1to1 or 1to2: each sender sends all weights
            weights_per_sender = {}
            for sender_idx in range(num_senders):
                weights_per_sender[sender_idx] = create_qwen3_full_model(config, device, dtype=torch.float16)
            total_params = sum(w.numel() for w in weights_per_sender[0].values()) * num_senders
            total_size_mb = sum(w.numel() * w.element_size() for w in weights_per_sender[0].values()) * num_senders / 1e6
            logger.info(f"[Receiver-{rank}] Allocated buffers for {num_senders} sender(s): {total_params/1e9:.2f}B params, {total_size_mb:.2f}MB")
        else:
            # 2to1 with weight slices: each sender sends different weights
            # Create one shared weight buffer and map slices to senders
            all_weights = create_qwen3_full_model(config, device, dtype=torch.float16)
            weight_items = list(all_weights.items())

            weights_per_sender = {}
            for sender_idx, (start_idx, end_idx) in enumerate(weight_slices):
                weights_per_sender[sender_idx] = dict(weight_items[start_idx:end_idx])

            total_params = sum(w.numel() for w in all_weights.values())
            total_size_mb = sum(w.numel() * w.element_size() for w in all_weights.values()) / 1e6
            logger.info(f"[Receiver-{rank}] Allocated shared buffer: {total_params/1e9:.2f}B params, {total_size_mb:.2f}MB")

        # Wait for all processes to be ready
        barrier.wait()

        all_transfer_times = []

        # Get sender metadata_session_ids (only once before all updates)
        # Collect all session IDs first, then sort by rank to ensure correct order
        session_id_list = []
        for _ in range(num_senders):
            sender_rank, session_id = session_info_queue.get(timeout=10)
            session_id_list.append((sender_rank, session_id))
            logger.info(f"[Receiver-{rank}] Got metadata_session_id from sender {sender_rank}: {session_id}")

        # Sort by sender rank to ensure correct mapping
        session_id_list.sort(key=lambda x: x[0])
        sender_session_ids = [session_id for _, session_id in session_id_list]

        for update_idx in range(num_updates):
            update_start = time.time()

            # Submit all transfer tasks
            all_handles = []
            submission_start = time.time()

            for sender_idx in range(num_senders):
                weights = weights_per_sender[sender_idx]
                sender_session_id = sender_session_ids[sender_idx]

                for weight_name, weight_tensor in weights.items():
                    ptr = weight_tensor.data_ptr()
                    length = weight_tensor.numel() * weight_tensor.element_size()

                    handle = receiver.submit_transfer_task(
                        name=weight_name,
                        metadata_session_id=sender_session_id,
                        ptr=ptr,
                        length=length,
                    )
                    all_handles.append((sender_idx, weight_name, handle))

            submission_end = time.time()

            logger.info(
                f"[Receiver-{rank}] Submitted {len(all_handles)} transfer tasks in "
                f"{(submission_end - submission_start)*1000:.2f}ms"
            )

            # Wait for all transfers to complete
            wait_start = time.time()
            for sender_idx, weight_name, handle in all_handles:
                handle.wait()

            torch.cuda.synchronize()
            wait_end = time.time()

            update_end = time.time()

            total_time = update_end - update_start
            wait_time = wait_end - wait_start
            total_bytes = sum(w.numel() * w.element_size() for w in weights_per_sender[0].values()) * num_senders

            all_transfer_times.append({
                'update_idx': update_idx,
                'total_time': total_time,
                'submission_time': submission_end - submission_start,
                'wait_time': wait_time,
                'num_tasks': len(all_handles),
                'num_senders': num_senders,
                'total_bytes': total_bytes,
            })

            logger.info(
                f"[Receiver-{rank}] Update {update_idx + 1}/{num_updates} completed: "
                f"total={total_time*1000:.2f}ms, submission={(submission_end - submission_start)*1000:.2f}ms, "
                f"wait={wait_time*1000:.2f}ms, bandwidth={(total_bytes * 8) / (total_time * 1e9):.2f}Gbps"
            )

            barrier.wait()

        avg_total_time = np.mean([t['total_time'] for t in all_transfer_times])
        avg_wait_time = np.mean([t['wait_time'] for t in all_transfer_times])
        total_bytes = all_transfer_times[0]['total_bytes']
        avg_bandwidth = (total_bytes * 8) / (avg_total_time * 1e9)

        stats = {
            'rank': rank,
            'num_updates': num_updates,
            'num_senders': num_senders,
            'avg_total_time': avg_total_time,
            'avg_wait_time': avg_wait_time,
            'avg_bandwidth_gbps': avg_bandwidth,
            'total_bytes': total_bytes,
            'thread_pool_size': receiver.thread_pool_size,
            'all_transfers': all_transfer_times,
        }

        result_queue.put((f"receiver_{rank}_stats", stats))

        logger.info(
            f"[Receiver-{rank}] All updates completed. Avg time: {avg_total_time*1000:.2f}ms, "
            f"Avg bandwidth: {avg_bandwidth:.2f}Gbps"
        )

    except Exception as e:
        logger.error(f"[Receiver-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"receiver_{rank}_error", str(e)))


# ============================================================================
# 1to1 Performance Test
# ============================================================================

def worker_process_1to1(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    session_info_queue: mp.Queue,
):
    """Entry point for 1to1 test worker."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        sender_process(rank, world_size, config, num_updates, result_queue, barrier, session_info_queue, "127.0.0.1", num_receivers=1)
    else:
        receiver_process(rank, world_size, config, num_updates, result_queue, barrier, session_info_queue, num_senders=1)


# ============================================================================
# 1to2 Performance Test
# ============================================================================

def worker_process_1to2(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    session_info_queue: mp.Queue,
):
    """Entry point for 1to2 test worker."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        sender_process(rank, world_size, config, num_updates, result_queue, barrier, session_info_queue, "127.0.0.1", num_receivers=world_size-1)
    else:
        receiver_process(rank, world_size, config, num_updates, result_queue, barrier, session_info_queue, num_senders=1)


# ============================================================================
# 2to1 Performance Test
# ============================================================================

def worker_process_2to1(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    session_info_queue: mp.Queue,
    num_senders: int,
    weight_slices: list,
):
    """Entry point for 2to1 test worker."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank < num_senders:
        # Each sender gets its corresponding weight slice
        # In 2to1, there's only 1 receiver (world_size - num_senders)
        sender_process(rank, world_size, config, num_updates, result_queue, barrier, session_info_queue, "127.0.0.1", weight_slice=weight_slices[rank], num_receivers=world_size-num_senders)
    else:
        receiver_process(rank, world_size, config, num_updates, result_queue, barrier, session_info_queue, num_senders=num_senders, weight_slices=weight_slices)


class TestP2PTransferPerformance(unittest.TestCase):
    """Performance tests for P2PTransferSender and P2PTransferReceiver."""

    def test_1to1_qwen3(self):
        """Test 1 sender + 1 receiver with Qwen3-32B."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        world_size = 2
        config = Qwen3_32BConfig()
        num_updates = 3

        logger.info(
            f"\n{'='*80}\n"
            f"Testing P2PTransferSender/Receiver (1to1)\n"
            f"Config: {world_size} processes, {num_updates} updates\n"
            f"{'='*80}\n"
        )

        result_queue = mp.Queue()
        session_info_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            worker_process_1to1,
            args=(world_size, config, num_updates, result_queue, barrier, session_info_queue),
            nprocs=world_size,
            join=False,
        )

        results = {}
        timeout = 120
        start_time = time.time()

        while len(results) < (world_size - 1):
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(10, remaining_time))
                results[key] = value
            except Exception:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        if "receiver_1_stats" in results:
            stats = results["receiver_1_stats"]
            logger.info(
                f"\n{'='*80}\n"
                f"P2PTransferSender/Receiver (1to1) Results:\n"
                f"  Thread pool size: {stats['thread_pool_size']}\n"
                f"  Avg total time: {stats['avg_total_time']*1000:.2f}ms\n"
                f"  Avg wait time: {stats['avg_wait_time']*1000:.2f}ms\n"
                f"  Avg bandwidth: {stats['avg_bandwidth_gbps']:.2f}Gbps\n"
                f"  Total data: {stats['total_bytes']/1e6:.2f}MB\n"
                f"{'='*80}\n"
            )

        result_queue.close()
        result_queue.join_thread()
        session_info_queue.close()
        session_info_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_1to2_qwen3(self):
        """Test 1 sender + 2 receivers with Qwen3-32B."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            self.skipTest("Requires at least 3 CUDA devices")

        world_size = 3
        config = Qwen3_32BConfig()
        num_updates = 2

        logger.info(
            f"\n{'='*80}\n"
            f"Testing P2PTransferSender/Receiver (1to2)\n"
            f"Config: {world_size} processes, {num_updates} updates\n"
            f"{'='*80}\n"
        )

        result_queue = mp.Queue()
        session_info_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            worker_process_1to2,
            args=(world_size, config, num_updates, result_queue, barrier, session_info_queue),
            nprocs=world_size,
            join=False,
        )

        results = {}
        timeout = 120
        start_time = time.time()

        while len(results) < (world_size - 1):
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(10, remaining_time))
                results[key] = value
            except Exception:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        logger.info(f"\n{'='*80}\nP2PTransferSender/Receiver (1to2) Results:\n")
        for rank in range(1, world_size):
            if f"receiver_{rank}_stats" in results:
                stats = results[f"receiver_{rank}_stats"]
                logger.info(
                    f"Receiver-{rank}: "
                    f"avg_time={stats['avg_total_time']*1000:.2f}ms, "
                    f"bandwidth={stats['avg_bandwidth_gbps']:.2f}Gbps, "
                    f"thread_pool_size={stats['thread_pool_size']}"
                )
        logger.info(f"{'='*80}\n")

        result_queue.close()
        result_queue.join_thread()
        session_info_queue.close()
        session_info_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_2to1_qwen3(self):
        """Test 2 senders + 1 receiver with Qwen3-32B."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            self.skipTest("Requires at least 3 CUDA devices")

        world_size = 3
        num_senders = 2
        config = Qwen3_32BConfig()
        num_updates = 2

        # Calculate weight slices: split weights between senders
        # Create a dummy model to get the total number of weights
        dummy_weights = create_qwen3_full_model(config, torch.device("cpu"), dtype=torch.float16)
        total_weights = len(dummy_weights)
        mid_point = total_weights // 2

        # Sender 0 gets first half, Sender 1 gets second half
        weight_slices = [(0, mid_point), (mid_point, total_weights)]

        logger.info(
            f"\n{'='*80}\n"
            f"Testing P2PTransferSender/Receiver (2to1)\n"
            f"Config: {world_size} processes ({num_senders} senders + 1 receiver), {num_updates} updates\n"
            f"Weight distribution: Sender-0: [0:{mid_point}], Sender-1: [{mid_point}:{total_weights}]\n"
            f"{'='*80}\n"
        )

        result_queue = mp.Queue()
        session_info_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            worker_process_2to1,
            args=(world_size, config, num_updates, result_queue, barrier, session_info_queue, num_senders, weight_slices),
            nprocs=world_size,
            join=False,
        )

        results = {}
        timeout = 120
        start_time = time.time()

        # Only waiting for receiver process
        while len(results) < 1:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(10, remaining_time))
                results[key] = value
            except Exception:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        if f"receiver_{num_senders}_stats" in results:
            stats = results[f"receiver_{num_senders}_stats"]
            logger.info(
                f"\n{'='*80}\n"
                f"P2PTransferSender/Receiver (2to1) Results:\n"
                f"  Thread pool size: {stats['thread_pool_size']}\n"
                f"  Number of senders: {stats['num_senders']}\n"
                f"  Avg total time: {stats['avg_total_time']*1000:.2f}ms\n"
                f"  Avg wait time: {stats['avg_wait_time']*1000:.2f}ms\n"
                f"  Avg bandwidth: {stats['avg_bandwidth_gbps']:.2f}Gbps\n"
                f"  Total data: {stats['total_bytes']/1e6:.2f}MB\n"
                f"{'='*80}\n"
            )

        result_queue.close()
        result_queue.join_thread()
        session_info_queue.close()
        session_info_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    unittest.main()
