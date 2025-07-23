import argparse

from vllm import EngineArgs, LLMEngine
from vllm.utils import FlexibleArgumentParser
import time
import torch

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch._inductor.config.compile_threads = 16

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""

    start = time.time()
    engine = initialize_engine(args)
    end = time.time()
    print(f"initialize_engine took {end - start:.2f} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(args)