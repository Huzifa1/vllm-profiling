## Setup

1) Clone the repo:
```bash
git clone https://github.com/Huzifa1/vllm-profiling.git
cd vllm-profiling
```

2) Install vllm with same version as the paper:
```bash
# Note that you need Python version 3.9 to 3.12.
pip install vllm==0.10.1.1
# Any version <=4.57.3
pip install transformers==4.57.3

pip install matplotlib
# Important to run Hybrid models (e.g., granite models)
pip install flashinfer-python
```

3) Apply custom vllm changes (addition of profiling logs):
```bash
python3 apply_vllm_changes.py
```

4) Download all models used in our experiments. This will download 23 LLMs. This will need around 0.5TB of disk space.
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
Where `<num>` is one of the following: '1', '2', '3-8', '9', '10', '11', '12', '13', '14', '16'.

You will find the figure in `figures/figure<num>/figure<num>.pdf`

