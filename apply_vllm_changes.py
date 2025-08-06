import shutil
import sys
from pathlib import Path

def apply_vllm_changes(vllm_path):
    vllm_dir = Path(vllm_path)
    
    target1 = vllm_dir / "model_executor/model_loader/base_loader.py"
    target2 = vllm_dir / "v1/engine/llm_engine.py"
    target3 = vllm_dir / "v1/worker/gpu_model_runner.py"

    shutil.copy2("vllm_files/base_loader.py", target1)
    shutil.copy2("vllm_files/llm_engine.py", target2)
    shutil.copy2("vllm_files/gpu_model_runner.py", target3)
    print(f"Changes applied!")

if __name__ == "__main__":
    try:
        import vllm
        import os
        vllm_path = os.path.dirname(vllm.__file__)
    except ImportError:
        print("Error: 'vllm' is not installed.")
        sys.exit(1)
        
    apply_vllm_changes(vllm_path)