## Setup

1) Clone the repo:
```bash
git clone https://github.com/Huzifa1/vllm-profiling.git
cd vllm-profiling
```

2) Install vllm with same version as the paper:
```bash
# Note that you need Python version 3.9 to 3.12.
# tensorize and runai are for different loading formats (Figure 13)
pip install "vllm[tensorizer,runai]==0.10.1.1"
# Any version <=4.57.3
pip install transformers==4.57.3

pip install matplotlib
# Important to run Hybrid models (e.g., granite models)
pip install flashinfer-python

pip install scikit-learn
```

3) Apply custom vllm changes (addition of profiling logs):
```bash
python3 apply_vllm_changes.py
```

4) Download all models used in our experiments. This will download multiple LLMs. This will need around 600GB of disk space.
```bash
python3 download_models.py <hf_token>
# `<hf_token>` is your HuggingFace token. Note that some models require applying to access before downloading.
```

## System Specifications

- In the following table, you can find the specification of our four nodes n1-n4.
- All experiments were run on node n1, with the following exceptions:
    - In Figure 10, we compare the results between n1 and n2.
    - In Figure 11 & 12, we compare the results between n1 and n3.

|        | node1 (n1)               | node2 (n2)               | node3 (n3)                                 | node4 (n4)                                 |
|--------|--------------------------|--------------------------|--------------------------------------------|--------------------------------------------|
| CPU    | AMD EPYC 9354 (32C)      | AMD EPYC 9354 (32C)      | 2× Intel Xeon Platinum 8568Y+ (2×48C)      | 2× Intel Xeon Gold 5520+ (2×28C)           |
| GPU    | H100 NVL                 | L40S                     | H100                                       | L40S                                       |
| DRAM   | DDR5 251GB               | DDR5 251GB               | DDR5 2TB                                   | DDR5 2TB                                   |
| OS     | Debian 12                | Debian 12                | Red Hat Enterprise Linux (RHEL) 9          | Red Hat Enterprise Linux (RHEL) 9          |
| Kernel | 6.1.0-40-amd64           | 6.1.0-40-amd64           | 5.14.0-503.34.1.el9_5                      | 5.14.0-503.34.1.el9_5                      |
| Python | 3.11.2                   | 3.11.2                   | 3.12                                       | 3.12                                       |
| CUDA   | 12.6 (Driver 580.82.07)  | 12.6 (Driver 580.82.07)  | 12.8 (Driver 570.124.06)                   | 12.8 (Driver 570.124.06)                   |
| PyTorch| 2.7.1+cu126              | 2.7.1+cu126              | 2.7.1+cu126                                | 2.7.1+cu126                                |
| vLLM   | v0.10.1.1                | v0.10.1.1                | v0.10.1.1                                  | v0.10.1.1                                  |
| SSD    | --                       | --                       | 4× PCIe 5.0 SSDs                           | --                                         |
| FS     | --                       | --                       | XFS/LVM with RAID-0 mirror                 | --                                         |


## Reproduce Figures

Note that even with identical environments, the results may still encounter +-5% variations

In order to reproduce any figure:
```bash
cd figures
bash run_figure.sh <num>
```
Where `<num>` is one of the following: '1', '2', '3-8', '7', '9', '10', '11', '12', '13', '14', '15', '17'.

You will find the figure in `figures/figure<num>/figure<num>.pdf`

### Important Notes

- Figures 3 to 8 (except 7) can all be generated with one experiment, therefore they are grouped as '3-8'
- Figure 10 requires running the experiemnts on 2 different GPUs. 
    - If you have 2 GPUs on the same machine, then you can do this by setting the environment variable `CUDA_VISIBLE_DEVICES` before running the `run_figure.sh` script. For example:
        ```bash
        # Run on first GPU
        CUDA_VISIBLE_DEVICES="0" bash run_figure.sh 10
        # Run on second GPU
        CUDA_VISIBLE_DEVICES="1" bash run_figure.sh 10
        ```
    - If you have 2 GPUs on different machines, then you can run the command `bash run_figure.sh 10` on each machine, then you need to move the output files of the second machine (e.g. `iterations`, `uncached` and `avg_comparison_results.json` files) to `./figures/figure10/gpu1`. Then to plot the figure, you can simply run:
        ```bash
        python3 figures/figure10/plot.py
        ```

