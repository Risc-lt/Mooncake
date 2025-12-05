import concurrent.futures
import logging
import os
import threading
import zmq

from queue import Empty, Queue
from typing import Dict, Optional

from mooncake.rl_p2p_transfer_manager import MooncakeTransferEngine, get_free_port

logger = logging.getLogger(__name__)


class TransferHandle:
    """A handle that mimics torch.distributed handle interface for compatibility"""

    def __init__(self, task_id: int, metadata_session_id: str):
        self.completed = False
        self.success = False
        self._event = threading.Event()
        self.task_id = task_id
        self.metadata_session_id = metadata_session_id

    def wait(self):
        """Wait for the transfer to complete"""
        self._event.wait()

        if not self.success:
            raise RuntimeError("P2P weight transfer failed")

    def _mark_done(self, success: bool):
        """Internal method to mark completion"""
        self.success = success
        self.completed = True
        self._event.set()


class TransferTask:
    """Represents a weight transfer task to be processed by workers"""

    def __init__(
        self,
        task_id: int,
        metadata_session_id: str,   # ZMQ communication endpoint (remote ip:port)
        transfer_session_id: str,   # Local Mooncake session_id for real RDMA transfer
        name: str,                  # Name for cache lookup
        ptr: int,
        length: int,
    ):
        self.task_id = task_id
        self.transfer_session_id = transfer_session_id
        self.name = name
        self.ptr = ptr
        self.length = length
        self.handle = TransferHandle(task_id, metadata_session_id)


class P2PTransferManagerBase:
    """
    Base class that maintains a single Mooncake TransferEngine with a thread pool.
    Derived classes implement their own task scheduling and processing logic.
    Similar to MooncakeKVManager architecture with single engine + multiple executors.
    """

    def __init__(self, hostname: str, gpu_id: int, pool_size: Optional[int] = None, ib_device: Optional[str] = None):
        """
        Initialize the P2P transfer manager base with a single engine and thread pool.

        Args:
            hostname: Local hostname for RDMA connections
            gpu_id: GPU device ID
            pool_size: Size of the thread pool (optional)
            ib_device: InfiniBand device name (optional)
        """
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = ib_device

        # Initialize single transfer engine (shared by all threads)
        self.engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )

        # Initialize thread pool for concurrent transfers
        cpu_count = os.cpu_count() or 8
        self.thread_pool_size = pool_size if pool_size is not None else min(max(2, int(0.25 * cpu_count)), 8)

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread_pool_size
        )

        # Task queue for all pending transfers
        self.task_queue: Queue = Queue()

        # Track registered memory regions per name
        # Structure: {name: ptr}
        self.registered_ptrs: Dict[str, int] = {}
        self.registered_ptrs_lock = threading.Lock()

        # Task counter for generating unique task IDs
        self.task_counter = 0
        self.task_counter_lock = threading.Lock()

        # Start the scheduler thread
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_worker,
            daemon=True
        )
        self.scheduler_thread.start()

        logger.info(
            f"{self.__class__.__name__} initialized with single engine and {self.thread_pool_size} worker threads "
            f"on {hostname}:{gpu_id}"
        )

    def _scheduler_worker(self):
        """
        Main scheduler thread that assigns tasks to thread pool workers.
        """
        raise NotImplementedError("Subclasses must implement _scheduler_worker")

    def _is_registered(self, name: str) -> bool:
        """
        Check if memory region is already registered.

        Args:
            name: Task name identifier

        Returns:
            True if registered, False otherwise
        """
        with self.registered_ptrs_lock:
            return name in self.registered_ptrs

    def _process_task(self, task: TransferTask, is_cached: bool = False):
        """
        Process a transfer task. Must be implemented by subclasses.

        Args:
            task: The transfer task
            is_cached: Whether the memory is already registered
        """
        raise NotImplementedError("Subclasses must implement _process_task")
    
    def batch_register_ptrs(self, names: list, ptrs: list, lens: list):
        """
        Batch register memory regions with the single engine.

        Args:
            names: List of task names
            ptrs: List of memory pointers
            lens: List of memory lengths
        """
        for name, ptr, length in zip(names, ptrs, lens):
            if self._is_registered(name):
                logger.debug(f"Ptr for name '{name}' already registered, skipping batch register")
                continue

            # Register the ptr with the single engine
            logger.debug(f"Batch registering ptr {ptr:#x} for name '{name}'")
            self.engine.register(ptr, length=length)
            self._update_registered_ptr(name, ptr)

    def _update_registered_ptr(self, name: str, ptr: int):
        """
        Update registered_ptrs cache after memory registration.

        Args:
            name: Task name identifier
            ptr: Memory pointer
        """
        with self.registered_ptrs_lock:
            # Add to cache if not already present
            if name not in self.registered_ptrs:
                self.registered_ptrs[name] = ptr
                logger.debug(f"Cached ptr {ptr:#x} for name '{name}'")

    def _remove_registered_ptr(self, name: str, ptr: int):
        """
        Remove from registered_ptrs cache after memory deregistration.

        Args:
            name: Task name identifier
            ptr: Memory pointer
        """
        with self.registered_ptrs_lock:
            if name in self.registered_ptrs:
                cached_ptr = self.registered_ptrs[name]
                if cached_ptr == ptr:
                    del self.registered_ptrs[name]
                    logger.debug(f"Removed cached ptr {ptr:#x} for name '{name}'")

    def _generate_task_id(self) -> int:
        """Generate a unique task ID"""
        with self.task_counter_lock:
            task_id = self.task_counter
            self.task_counter += 1
        return task_id

    def get_registered_ptr(self, name: str) -> Optional[int]:
        """
        Get the registered pointer for a given name.

        Returns:
            int: ptr for the given name, or None if not found
        """
        with self.registered_ptrs_lock:
            if name not in self.registered_ptrs:
                logger.debug(f"No registered ptr found for name '{name}' (not yet cached)")
                return None
            return self.registered_ptrs[name]

class P2PTransferReceiver(P2PTransferManagerBase):
    """
    Receiver-side P2P transfer manager.

    The receiver:
    1. Registers local memory regions
    2. Sends sync_status to sender (training side) via ZMQ
    3. Waits for transfer completion confirmation

    Uses the base class cache-aware scheduler for optimal engine allocation.
    """

    def __init__(self, hostname: str, gpu_id: int, pool_size: Optional[int] = None, ib_device: Optional[str] = None):
        """Initialize receiver with task queue and scheduler."""
        # Set default pool size to 1 if not provided
        super().__init__(hostname, gpu_id, pool_size, ib_device)

        logger.info(f"P2PTransferReceiver initialized")

    def _scheduler_worker(self):
        """
        Main scheduler thread that submits tasks to thread pool for processing.
        """
        while self.scheduler_running:
            try:
                # Block until a task is available
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    continue

                # Check if memory is already registered
                is_cached = self._is_registered(task.name)

                logger.debug(
                    f"Scheduler submitting task {task.task_id} (name={task.name}) to thread pool, "
                    f"cached={is_cached}"
                )

                # Submit task to thread pool for processing
                self.executor.submit(self._process_task, task, is_cached)

            except Empty:
                # Expected timeout when no tasks are available, ignore silently
                continue
            except Exception as e:
                logger.error(f"Scheduler worker error: {e}", exc_info=True)

    def _process_task(self, task: TransferTask, is_cached: bool = False):
        """
        Process a receiver-side transfer task.

        Steps:
        1. Register memory region if not already cached
        2. Send sync_status to training side
        3. Wait for confirmation
        4. Mark handle as done
        """
        try:
            # Get the engine's session_id (single engine)
            local_transfer_session_id = self.engine.get_session_id()
            task.transfer_session_id = local_transfer_session_id

            # Check if memory is already registered (cached)
            if is_cached:
                logger.debug(
                    f"Using cached registered memory for task {task.task_id}: "
                    f"ptr={task.ptr:#x}, length={task.length}"
                )
            else:
                # Register memory region with the single engine
                logger.debug(
                    f"Registering memory for task {task.task_id}: "
                    f"ptr={task.ptr:#x}, length={task.length}"
                )
                self.engine.register(task.ptr, task.length)
                self._update_registered_ptr(task.name, task.ptr)

            # Send sync_status and wait for confirmation
            self._send_sync_status_and_wait(
                task=task,
                metadata_session_id=task.handle.metadata_session_id,
                transfer_session_id=local_transfer_session_id,
            )

            # Mark as successfully completed
            task.handle._mark_done(True)

        except Exception as e:
            logger.error(
                f"Receiver task {task.task_id} failed: {e}",
                exc_info=True
            )
            task.handle._mark_done(False)

    def _send_sync_status_and_wait(
        self,
        task: TransferTask,
        metadata_session_id: str,
        transfer_session_id: str,
    ):
        """
        Send sync_status message to training side and wait for confirmation.
        """
        # Create a temporary socket
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.connect(f"tcp://{metadata_session_id}")

        try:
            # Send sync_status to training side
            # DEALER sends [empty_delimiter, message] which ROUTER receives as [identity, empty_delimiter, message]
            socket.send_multipart([
                b"",  # Empty delimiter frame
                zmq.utils.jsonapi.dumps({
                    "type": "sync_status",
                    "metadata_session_id": metadata_session_id,
                    "status": "ready",
                    "ip": self.hostname,
                    "transfer_session_id": transfer_session_id,
                    "name": task.name,
                    "ptr": task.ptr,
                    "length": task.length,
                    "task_id": task.task_id,
                })
            ])
            logger.info(
                f"Sent sync_status to {metadata_session_id} for session {metadata_session_id}, "
                f"task_id={task.task_id}, transfer_session_id={transfer_session_id}, ptr={task.ptr:#x}"
            )

            # Wait for confirmation from training side
            socket.setsockopt(zmq.RCVTIMEO, 60000)  # 60 seconds timeout

            logger.debug(f"Waiting for confirmation from {metadata_session_id} for task {task.task_id}")

            try:
                frames = socket.recv_multipart()
                logger.debug(f"Received {len(frames)} frames: {[len(f) for f in frames]}")

                # Find the JSON payload (skip empty frames)
                response = None
                for i, frame in enumerate(frames):
                    if len(frame) > 0:
                        try:
                            response = zmq.utils.jsonapi.loads(frame)
                            logger.debug(f"Parsed JSON from frame {i}: {response}")
                            break
                        except Exception:
                            logger.debug(f"Frame {i} is not JSON: {repr(frame[:100])}")
                            continue

                if response is None:
                    raise RuntimeError(f"No valid JSON found in {len(frames)} frames")

            except Exception as json_error:
                logger.error(f"Failed to receive/parse response: {json_error}")
                raise json_error

            # Check response status
            response_type = response.get("type", "")
            response_status = response.get("status", "")

            if response_type == "transfer_complete" and response_status == "success":
                logger.info(
                    f"Received success confirmation from {metadata_session_id} "
                    f"for task {task.task_id}, transfer session {transfer_session_id}"
                )
                # Clean up the socket after successful transfer
                socket.close()
                context.term()
                logger.debug(f"Released socket for task {task.task_id}")
            else:
                error_msg = response.get("error", "Unknown error")
                logger.error(
                    f"Received failure confirmation from {metadata_session_id} "
                    f"for task {task.task_id}: {error_msg}"
                )
                raise RuntimeError(
                    f"Transfer failed for task {task.task_id}: {error_msg}"
                )

        except zmq.Again:
            # Timeout waiting for response
            logger.error(
                f"Timeout waiting for confirmation from {metadata_session_id} "
                f"for task {task.task_id}"
            )
            # Clean up socket on timeout
            socket.close()
            context.term()
            raise RuntimeError(
                f"Timeout waiting for transfer confirmation for task {task.task_id}"
            )
        except Exception as e:
            # If sending or receiving fails, clean up the socket immediately
            logger.error(
                f"Error in sync_status_and_wait for task {task.task_id}: {e}"
            )
            socket.close()
            context.term()
            raise e

    def submit_transfer_task(
        self, name: str, metadata_session_id: str, ptr: int, length: int
    ) -> TransferHandle:
        """
        Submit a transfer task to the queue for processing (receiver side).
        Returns a handle that can be used to wait for completion.

        The task will be queued and assigned to an engine by the cache-aware scheduler.

        Args:
            name: Task name identifier for cache tracking
            session_id: Remote training process address (ip:port) for ZMQ communication
            ptr: Local memory pointer to register
            length: Buffer length
        """
        task_id = self._generate_task_id()
        try:
            # Validate metadata_session_id format
            _ = metadata_session_id.split(":")[0]
            _ = int(metadata_session_id.split(":")[1])
        except Exception as e:
            raise ValueError(f"Invalid metadata_session_id format: {metadata_session_id}") from e

        # Create task object
        task = TransferTask(
            task_id=task_id,
            metadata_session_id=metadata_session_id,    # remote ZMQ communication endpoint
            transfer_session_id="",                     # auto set by engine
            name=name,
            ptr=ptr,
            length=length,
        )

        # Put task into queue - scheduler will assign it to best engine
        self.task_queue.put(task)

        logger.debug(
            f"Queued transfer task {task_id} (name={name}) for session {metadata_session_id}, "
            f"ptr={ptr:#x}, length={length}"
        )

        return task.handle


class P2PTransferSender(P2PTransferManagerBase):
    """
    Sender-side P2P transfer manager.

    Architecture:
    1. batch_register_ptrs: Pre-register RDMA memory regions
    2. Single ZMQ.ROUTER listening for multiple receiver DEALER connections
    3. Scheduler receives messages and performs RDMA writes
    4. Sends completion signals back through the same ROUTER
    """

    def __init__(self, hostname: str, gpu_id: int, pool_size: Optional[int] = None,
                 ib_device: Optional[str] = None):
        """
        Initialize the sender-side P2P transfer manager.

        Args:
            hostname: Local hostname for RDMA connections
            gpu_id: GPU device ID
            pool_size: Size of the engine pool (optional)
            ib_device: InfiniBand device name (optional)
        """
        # Create ZMQ ROUTER socket BEFORE calling super().__init__()
        # because super().__init__() starts the scheduler thread which needs self.socket
        self.zmq_port = get_free_port()
        self.metadata_session_id = f"{hostname}:{self.zmq_port}"

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://0.0.0.0:{self.zmq_port}")

        # Set socket timeout for non-blocking receive
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        # Socket lock for thread-safe sending (multiple threads in pool send responses)
        self.socket_lock = threading.Lock()

        # Now call super().__init__() which will start the scheduler thread
        super().__init__(hostname, gpu_id, pool_size, ib_device)

        logger.info(f"P2PTransferSender initialized, listening on {self.metadata_session_id}")

    def __del__(self):
        """Clean up ZMQ resources on deletion"""
        try:
            self.socket.close()
            self.context.term()
            logger.debug("Cleaned up ZMQ resources in P2PTransferSender destructor")
        except Exception as e:
            logger.error(f"Error cleaning up ZMQ resources in destructor: {e}")

    def _scheduler_worker(self):
        """
        Main scheduler thread that listens on ZMQ ROUTER and processes transfer requests.

        Workflow:
        1. Listen for sync_status messages from receiver DEALERs
        2. Find cached engine for the memory region
        3. Perform RDMA write
        4. Send completion signal back through ROUTER
        """
        while self.scheduler_running:
            try:
                # Receive message from ROUTER (non-blocking with timeout)
                try:
                    frames = self.socket.recv_multipart()
                except zmq.Again:
                    # Timeout, no messages available
                    continue

                # ROUTER frames: [identity, empty_delimiter, message]
                if len(frames) < 3:
                    logger.warning(f"Received malformed message with {len(frames)} frames")
                    continue

                identity = frames[0]
                message = zmq.utils.jsonapi.loads(frames[2])

                if message.get("type") != "sync_status":
                    logger.warning(f"Received unexpected message type: {message.get('type')}")
                    continue

                # Extract transfer details
                name = message["name"]
                remote_ptr = message["ptr"]
                remote_transfer_session_id = message["transfer_session_id"]
                length = message["length"]
                task_id = message["task_id"]

                logger.info(
                    f"Received sync_status for task {task_id}: "
                    f"name={name}, remote_ptr={remote_ptr:#x}, length={length}"
                )

                # Find the cached memory region
                local_ptr = self.get_registered_ptr(name)
                if local_ptr is None:
                    error_msg = f"Memory region '{name}' not registered on sender"
                    logger.error(error_msg)

                    # Send error response (thread-safe)
                    with self.socket_lock:
                        self.socket.send_multipart([
                            identity,
                            b"",
                            zmq.utils.jsonapi.dumps({
                                "type": "transfer_complete",
                                "status": "failed",
                                "error": error_msg,
                                "task_id": task_id,
                            })
                        ])
                    continue

                logger.debug(f"Submitting task {task_id} (name={name}) to thread pool")

                # Submit RDMA write to thread pool
                self.executor.submit(
                    self._process_transfer,
                    identity=identity,
                    task_id=task_id,
                    name=name,
                    local_ptr=local_ptr,
                    remote_ptr=remote_ptr,
                    remote_transfer_session_id=remote_transfer_session_id,
                    length=length,
                )

            except Exception as e:
                logger.error(f"Scheduler worker error: {e}", exc_info=True)

    def _process_transfer(
        self,
        identity: bytes,
        task_id: int,
        name: str,
        local_ptr: int,
        remote_ptr: int,
        remote_transfer_session_id: str,
        length: int,
    ):
        """
        Process a single transfer: perform RDMA write and send completion signal.
        """
        try:
            # Perform RDMA write using the single engine
            logger.info(
                f"Starting RDMA write for task {task_id}: "
                f"local_ptr={local_ptr:#x} -> remote_ptr={remote_ptr:#x}, length={length}"
            )

            ret = self.engine.transfer_sync(
                session_id=remote_transfer_session_id,
                buffer=local_ptr,
                peer_buffer_address=remote_ptr,
                length=length,
            )

            if ret < 0:
                error_msg = f"RDMA write failed for task {task_id}"
                logger.error(error_msg)

                # Send failure confirmation (thread-safe)
                with self.socket_lock:
                    self.socket.send_multipart([
                        identity,
                        b"",
                        zmq.utils.jsonapi.dumps({
                            "type": "transfer_complete",
                            "status": "failed",
                            "error": error_msg,
                            "task_id": task_id,
                        })
                    ])
                return

            logger.info(f"RDMA write completed successfully for task {task_id}")

            # Send success confirmation (thread-safe)
            with self.socket_lock:
                self.socket.send_multipart([
                    identity,
                    b"",
                    zmq.utils.jsonapi.dumps({
                        "type": "transfer_complete",
                        "status": "success",
                        "task_id": task_id,
                    })
                ])

            logger.info(f"Sent transfer_complete confirmation for task {task_id}")

        except Exception as e:
            logger.error(f"Transfer processing failed for task {task_id}: {e}", exc_info=True)
