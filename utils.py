import torch
import subprocess

def clear_cache():
    # Clear cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Execute a shell command to clear system caches
    subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, check=True)
    
def num_of_batch_sizes(cuda_graph_sizes):
    return len([1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)])