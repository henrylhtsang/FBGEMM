# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import logging

import torch

from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import (
    cutlass_blackwell_fmha_interface as fmha,
)

torch.autograd.set_multithreading_enabled(False)
torch.use_deterministic_algorithms(True)

"""
Run:

buck2 run -c fbcode.nvcc_arch=b200a -c fbcode.platform010_cuda_version=12.8 @mode/opt //scripts/henrylhtsang/repros:fmha2  2>&1 | tee output.log

"""

torch.manual_seed(0)


def main() -> None:
    for i in range(1000):
        torch.manual_seed(2)
        logging.warning(f"Iteration {i + 1}/100")

        # Forward inputs
        q = torch.randn((47, 4, 128), dtype=torch.bfloat16, device="cuda:0")
        k = torch.randn((471, 4, 128), dtype=torch.bfloat16, device="cuda:0")
        v = torch.randn((471, 4, 128), dtype=torch.bfloat16, device="cuda:0")
        cu_seqlens_q = torch.tensor(
            [0, 10, 44, 45, 46, 47], device="cuda:0", dtype=torch.int32
        )
        cu_seqlens_k = torch.tensor(
            [0, 259, 380, 394, 446, 471], device="cuda:0", dtype=torch.int32
        )
        max_seq_len_q = 34
        max_seq_len_k = 259

        # Forward pass
        out, lse = fmha._cutlass_blackwell_fmha_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            bottom_right=False,
        )

        # # Backward pass
        # dout = torch.randn_like(out)
        # dq, dk, dv = fmha._cutlass_blackwell_fmha_backward(
        #     dout=dout,
        #     q=q,
        #     k=k,
        #     v=v,
        #     out=out,
        #     softmax_lse=lse,
        #     cu_seqlens_q=cu_seqlens_q,
        #     cu_seqlens_k=cu_seqlens_k,
        #     max_seq_len_q=max_seq_len_q,
        #     max_seq_len_k=max_seq_len_k,
        # )

    print("Done - all 100 iterations completed successfully")


if __name__ == "__main__":
    main()
