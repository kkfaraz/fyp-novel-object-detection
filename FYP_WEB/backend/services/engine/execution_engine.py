import logging
from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from backend.services.utils.env_manager import EnvironmentManager

logger = logging.getLogger("ExecutionEngine")

class ExecutionEngine:
    def __init__(self):
        self.env = EnvironmentManager()
        self.num_workers = self.env.get_workers()
        # Disable parallel execution to save VRAM and avoid concurrency-based CUDA OOMs
        self.use_parallel = False
        logger.info(f"ExecutionEngine initialized: use_parallel={self.use_parallel}, workers={self.num_workers}")

    def run_tasks(self, tasks: List[Callable[[], Any]]) -> List[Any]:
        """
        Executes a list of task functions sequentially or concurrently.
        """
        if not tasks:
            return []

        if not self.use_parallel:
            # Sequential execution
            import torch
            results = []
            for i, task in enumerate(tasks):
                try:
                    results.append(task())
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Task {i} failed: {e}")
                    raise
            return results
        
        # Concurrent execution using ThreadPoolExecutor
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.num_workers)) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(task): idx for idx, task in enumerate(tasks)}
            
            for future in future_to_index:
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Concurrent task {idx} failed: {e}")
                    raise
                    
        return results
