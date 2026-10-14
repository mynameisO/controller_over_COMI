"""Exercise BF16 CUDA kernels before spending time loading model weights."""
import argparse


def check_cuda(num_gpus=1):
    import torch
    if num_gpus < 1 or torch.cuda.device_count() < num_gpus:
        raise SystemExit("Requested GPU count is unavailable")
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("Compiled architectures:", torch.cuda.get_arch_list())
    for index in range(num_gpus):
        with torch.cuda.device(index):
            properties = torch.cuda.get_device_properties(index)
            print(index, properties.name, "capability", torch.cuda.get_device_capability(index),
                  round(properties.total_memory / 2**30, 1), "GiB")
            if not torch.cuda.is_bf16_supported():
                raise SystemExit(f"GPU {index} does not support BF16")
            try:
                x = torch.randn(64, 64, device=f"cuda:{index}",
                                dtype=torch.bfloat16, requires_grad=True)
                (x @ x).float().square().mean().backward()
                # Exercise embedding and attention kernels used by the model too.
                indices = torch.arange(8, device=x.device)
                torch.nn.functional.embedding(indices, x)
                q = x.detach().view(1, 1, 64, 64)
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize(index)
                del x, q, indices
            except RuntimeError as error:
                raise SystemExit(
                    f"CUDA kernel check failed on GPU {index}: {error}\n"
                    "For Blackwell, install the prepared torch==2.8.0/cu129 stack "
                    "and verify the Pod's NVIDIA driver is compatible."
                ) from error
            print("BF16 forward/backward, embedding and SDPA: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, default=1)
    check_cuda(parser.parse_args().num_gpus)
