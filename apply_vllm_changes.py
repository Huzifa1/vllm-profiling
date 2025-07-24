import shutil
import sys
from pathlib import Path

def apply_vllm_changes(vllm_path):
    vllm_dir = Path(vllm_path)
    
    target1 = vllm_dir / "model_executor/model_loader/base_loader.py"
    target2 = vllm_dir / "v1/engine/llm_engine.py"

    shutil.copy2("vllm_files/base_loader.py", target1)
    shutil.copy2("vllm_files/llm_engine.py", target2)
    print(f"Changes applied!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python apply_vllm_changes.py <vllm_path>")
        sys.exit(1)
    vllm_path = sys.argv[1]
    apply_vllm_changes(vllm_path)