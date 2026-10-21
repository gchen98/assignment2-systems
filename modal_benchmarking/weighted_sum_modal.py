from pathlib import Path
from typing import Optional
import modal
import torch.profiler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create a console handler
console_handler = logging.StreamHandler()
# Create a formatting layout
formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
# Add the handler to your logger
logger.addHandler(console_handler)

traces = modal.Volume.from_name("cs336_traces",create_if_missing=True)
TRACE_DIR = Path("/traces")

image = (modal.
         Image.debian_slim()
         .uv_sync(uv_project_dir="./",
                  extra_options="--no-install-package cs336-basics")
         .add_local_python_source("cs336_basics")
         .add_local_python_source("cs336_systems")
         )

gpu="A100-80GB"
app = modal.App("model-benchmarking")
config = {"gpu":gpu, "image":image}

@app.function(image = image,gpu=gpu)
def weighted_sum_comparison():
    def make_dummy(batch_size, dimension):
        x_list = []
        for j in range(batch_size):
            row = []
            for i in range(dimension):
                # print(f" {j * 10 + i}", end='')
                row.append(j * 10 + i)
            x_list.append(row)
            # print()
        x = torch.tensor(x_list, dtype=torch.float32)
        weight = torch.arange(dimension, dtype=torch.float32)
        return (x, weight)

    batch_size = 40
    dimension = 50

    (x, weight) = make_dummy(batch_size, dimension)
    rand_grad = torch.rand(batch_size).to("cuda")

    x_ref = x.clone().detach().to("cuda").requires_grad_()
    weight_ref = weight.clone().detach().to("cuda").requires_grad_()

    x_custom = x.clone().detach().to("cuda").requires_grad_()
    weight_custom = weight.clone().detach().to("cuda").requires_grad_()

    from einops import einsum
    y_ref = einsum(x_ref, weight_ref, "... batch dim, dim -> ... batch")
    logger.info(f"reference forward {y_ref} with shape {y_ref.shape}")
    y_ref.backward(rand_grad)
    logger.info(f"reference gradients {x_ref.grad} {weight_ref.grad}")

    from cs336_systems.weighted_sum import WeightedSumFunc
    y_custom = WeightedSumFunc.apply(x_custom, weight_custom)
    logger.info(f"triton forward {y_custom}")
    y_custom.backward(rand_grad)
    logger.info(f"triton gradients {x_custom.grad} {weight_custom.grad}")

@app.function(image = image,gpu=gpu)
def benchmark_triton():
    def make_dummy(batch_size, dimension):
        x_list = []
        for j in range(batch_size):
            row = []
            for i in range(dimension):
                # print(f" {j * 10 + i}", end='')
                row.append(j * 10 + i)
            x_list.append(row)
            # print()
        x = torch.tensor(x_list, dtype=torch.float32)
        weight = torch.arange(dimension, dtype=torch.float32)
        return (x, weight)

    batch_size = 40
    dimension = 50

    (x, weight) = make_dummy(batch_size, dimension)
    rand_grad = torch.rand(batch_size).to("cuda")

    x_ref = x.clone().detach().to("cuda").requires_grad_()
    weight_ref = weight.clone().detach().to("cuda").requires_grad_()

    x_custom = x.clone().detach().to("cuda").requires_grad_()
    weight_custom = weight.clone().detach().to("cuda").requires_grad_()

    def forward_backward_ref():
        from einops import einsum
        y_ref = einsum(x_ref, weight_ref, "... batch dim, dim -> ... batch")
        # logger.info(f"reference forward {y_ref} with shape {y_ref.shape}")
        y_ref.backward(rand_grad)
        # logger.info(f"reference gradients {x_ref.grad} {weight_ref.grad}")

    from cs336_systems.weighted_sum import WeightedSumFunc
    def forward_backward_triton():
        y_custom = WeightedSumFunc.apply(x_custom, weight_custom)
        # logger.info(f"triton forward {y_custom}")
        y_custom.backward(rand_grad)
    # logger.info(f"triton gradients {x_custom.grad} {weight_custom.grad}")
    import triton
    ms = triton.testing.do_bench(forward_backward_ref)
    logger.info(f"Mean execution time forward backward ref is {ms}")

    ms = triton.testing.do_bench(forward_backward_triton)
    logger.info(f"Mean execution time forward backward triton is {ms}")


@app.local_entrypoint()
def main():
    # weighted_sum_comparison.remote()
    benchmark_triton.remote()

if __name__=="__main__":
    main()
