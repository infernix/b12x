# Lossless BF16 PCIe two-shot collective qualification

Status: **qualified** for four-GPU tensor-parallel GLM-5.3-Flash decode on
NVIDIA RTX PRO 6000 Blackwell GPUs (SM120).

The `PCIeTwoShotBF16` collective transfers BF16 payloads over CUDA peer memory,
accumulates values in FP32 in a fixed rank order, and rounds the result to BF16
once. Its public operations are reduce-scatter, all-gather, and all-reduce. The
implementation rejects overlapping input and output storage because its
non-coherent peer loads cannot safely observe storage written by the same
kernel launch.

## Correctness contract

The distributed qualification test is:

```bash
NCCL_ALGO=Ring CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.run --nproc-per-node=4 \
  tests/comm/test_pcie_twoshot_bf16.py
```

The test executes all three public collectives at multiple tensor heights. It
checks the reduction against an exact FP32 sum, requires at most one BF16
rounding, verifies deterministic eager execution, rejects overlapping storage,
and captures and replays all-reduce in a CUDA graph. The graph owns its output
allocation after capture; replay must retain the output address and must not
increase PyTorch CUDA allocator usage.

## GLM-5.3-Flash serving measurement

The measured B12X integration source was commit
`cd89e4c7cf36e3366e49b0c09ef5e2deed45b8ea` with Git tree
`7f7107f7973a65aa5a94162b12a80b430abee252`. The measured vLLM integration
source was commit `c057c05522ca4b158be97a22a935633a00506124` with Git tree
`171465307585ecae2319284fe72d3a67610c5998`. These identifiers define the
software boundary for the measurements below; the enabled and disabled arms
used the same source trees.

The serving comparison used physical GPUs 4–7 at stock clocks, tensor
parallelism 4, decode context parallelism 1, an NVFP4 target checkpoint, an
MXFP8 DFlash2 draft checkpoint, seven speculative tokens per verifier step,
FP8 KV cache, B12X attention/MoE/linear kernels, and identical source and launch
arguments in both arms. The control disabled this collective with
`VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=0`; the candidate enabled it through
768 KiB. Each concurrency cell used a 15-second warmup followed by three
30-second samples. `C8` and `C12` mean 8 and 12 concurrent requests.

| Arm | Concurrency | Output tok/s samples | Verifier steps/s samples | Median output tok/s | Median verifier steps/s |
|:--|--:|:--|:--|--:|--:|
| Disabled | 8 | 732.047, 731.156, 729.200 | 281.098, 280.580, 282.060 | 731.156 | 281.098 |
| Enabled | 8 | 754.231, 748.416, 753.032 | 285.293, 286.877, 293.970 | 753.032 | 286.877 |
| Disabled | 12 | 865.016, 897.032, 900.058 | 341.857, 338.963, 342.022 | 897.032 | 341.857 |
| Enabled | 12 | 906.205, 909.330, 896.085 | 350.595, 351.172, 348.355 | 906.205 | 350.595 |

The enabled-minus-disabled median change, calculated as
`(enabled / disabled - 1) * 100`, was:

- C8: **+2.99% output tok/s** and **+2.06% verifier steps/s**.
- C12: **+1.02% output tok/s** and **+2.56% verifier steps/s**.

The conclusion is limited to the declared four-GPU SM120 topology and tensor
sizes selected by the vLLM integration. Other GPU architectures, world sizes,
and message sizes remain unsupported until separately qualified.
