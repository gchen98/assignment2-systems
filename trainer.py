from pathlib import Path
from typing import Optional
import modal
import torch.profiler

traces = modal.Volume.from_name("cs336_traces",create_if_missing=True)
TRACE_DIR = Path("/traces")

image = (modal.
         Image.debian_slim()
         .uv_sync(uv_project_dir="./",
                  extra_options="--no-install-package cs336-basics")
         .add_local_python_source("cs336_basics"))

gpu="A100-80GB"
app = modal.App("model-benchmarking")
config = {"gpu":gpu, "image":image}

@app.local_entrypoint()
def main(
        function: str = "underutilize",
        label: Optional[str] = None,
        steps: int = 3,
        schedule=None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        print_rows: int = 10,
        kwargs_json_path: Optional[str] = None,
        device: str="cuda",
        training_mode:str="training",
        model_params:str,
        batch_size:int=4,
        vocab_size:int=10000,
        steps:int,
        warmup_steps:int,
):
    if kwargs_json_path is not None:  # use to pass arguments to function
        import json
        kwargs = json.loads(Path(kwargs_json_path).read_text())
    else:
        kwargs = {}


    results, remote_path = profile.remote(
        function,
        label=label,
        steps=steps,
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        print_rows=print_rows,
        **kwargs,
    )

    output_path = Path("/tmp") / remote_path.name
    output_path.write_text(results)
    print(f"trace saved locally at {output_path}")

def main2(*args:str):
    import sys
    import argparse
    import yaml
    import numpy as np
    print(f"args is {args}",file=sys.stderr)
    parser = argparse.ArgumentParser(description="A language model trainer")
    parser.add_argument("--device", required=True, help="device to run on")
    parser.add_argument("--training_mode", required=True, help="'forward' or 'training'")
    parser.add_argument("--model_params", required=True, help="YAML file for model parameters")
    parser.add_argument("--batch_size", required=True, help="number of training examples to sample randomly")
    parser.add_argument("--vocab_size",required=True, help="number of vocab tokens")
    parser.add_argument("--steps", required=True, help="optimization steps", default=1000)
    parser.add_argument("--max_lr", required=True, help="maximum learning rate")
    parser.add_argument("--min_lr", required=True, help="minimum learning rate")
    parser.add_argument("--warmup_steps", required=True, help="warm up steps to take before cosine decay learning rate")
    parser.add_argument("--max_gradient_norm", required=True, help="L2 norm of gradients threshold for clipping.")

    args = parser.parse_args(args = args)

    model_params = None
    with open(args.model_params, "r") as f_in:
        yaml_data = yaml.safe_load(f_in)
        model_params = yaml_data['model_params']

    if model_params is None:
        print(f"Could not load model config file {args.model_params}",file=sys.stderr)
        sys.exit(1)

    sanity_check.remote(args)
    metrics = benchmarker.remote(args,model_params)
    # metrics = benchmarker(args,model_params)
    f_times = []
    b_times = []
    o_times = []
    for metric in metrics:
        if metric['warmed_up']:
            f_times.append(metric['forward_time'])
            b_times.append(metric['backward_time'])
            o_times.append(metric['opt_time'])
    print(f"mean forward time {np.array(f_times).mean()} backward time {np.array(b_times).mean()} optimizer time {np.array(o_times).mean()}")
    print(f"var forward time {np.array(f_times).var()} backward time {np.array(b_times).var()} optimizer time {np.array(o_times).var()}")



@app.function(image = image, gpu=gpu)
def sanity_check(args):
    import torch
    print(torch.cuda.get_device_name("cuda"))
    print(torch.cuda.get_device_properties("cuda"))

@app.function(volumes={TRACE_DIR: traces}, **config)
def profile(
        function_name: str,
        label: Optional[str] = None,
        steps: int = 3,
        schedule = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        print_rows: int = 0,
        **kwargs,
):
    from uuid import uuid4
    try:
        function = app.registered_functions[function_name]
    except KeyError:
        raise ValueError(f"Function {function_name} not found!")

    output_dir = (TRACE_DIR/(function_name+(f"_{label}" if label else "")) / str(uuid4()))
    output_dir.mkdir(parents = True, exist_ok= True)

    if schedule is None:
        if steps < 3:
            raise ValueError("Steps must be at least 3 when using default schedule")
        schedule = {"wait":1, "warmup": 1, "active": steps - 2, "repeat": 0 }
    schedule = torch.profiler.schedule(**schedule)

    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule = schedule,
        record_shapes = record_shapes,
        profile_memory = profile_memory,
        with_stack = with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        # add the initialization steps here



        for _ in range(steps):
            function.local(**kwargs)
            prof.step()
    if print_rows:
        print(
            prof.key_averages().table(sort_by="cuda_time_total",row_limit=print_rows)
        )
    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key = lambda pth:pth.stat().st_mtime, reverse = True,
    )[0]

    print(f"trace saved to {trace_path.relative_to((TRACE_DIR))}")
    return trace_path.read_text(),trace_path.relative_to(TRACE_DIR)



@app.function(image = image, gpu=gpu)
def benchmarker(args,model_params)->list[dict]:
    # import sys
    import time
    import timeit
    import logging
    import torch
    import cs336_basics
    from cs336_basics import model, data, nn_utils, optimizer

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create a console handler
    console_handler = logging.StreamHandler()
    # Create a formatting layout
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    # Add the handler to your logger
    logger.addHandler(console_handler)
    def get_batch(vocab_size: int, batch_size: int, context_length: int, device: str) -> torch.Tensor:
        a = torch.randint(0, vocab_size, (batch_size, context_length + 1))
        x = a[:, :-1]
        y = a[:, 1:]
        if "cuda" in device:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    def get_time(device: str) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return timeit.default_timer()

    training_mode = args.training_mode
    batch_size = int(args.batch_size)
    warmup_steps = int(args.warmup_steps)
    device = args.device
    context_length = model_params['context_length']
    embedding_dim = model_params['embedding_dim']
    num_layers = model_params['num_layers']
    num_heads = model_params['num_heads']
    feed_forward_dim = model_params['feed_forward_dim']
    rope_theta = model_params['rope_theta']
    steps = int(args.steps)
    # get the vocab size
    vocab_size = int(args.vocab_size)
    # create the language model
    logger.debug("Creating a transformer")
    # return
    transformer_model = model.BasicsTransformerLM( vocab_size = vocab_size,
                                                   context_length = context_length,
                                                   d_model = embedding_dim,
                                                   num_layers = num_layers,
                                                   num_heads = num_heads,
                                                   d_ff = feed_forward_dim,
                                                    rope_theta = rope_theta).to(device)
    model_size:int = 0
    for name, buf in transformer_model.named_buffers():
        num_elements = buf.numel()
        shape = buf.shape
        dtype = buf.dtype
        element_size = buf.element_size()
        model_size+=num_elements*element_size
        logger.debug(f"buffer of {dtype} with shape {shape} and elements {num_elements} of size {element_size}")
        model_size += num_elements * element_size
    for param in transformer_model.parameters(recurse=True):
        num_elements = param.numel()
        shape = param.shape
        dtype = param.dtype
        element_size = param.element_size()
        logger.debug(f"param of {dtype} with shape {shape} and elements {num_elements} of size {element_size}")
        model_size+=num_elements*element_size
    mb = model_size/ (1024**2)
    logger.info(f"Model size is {model_size} bytes, or {mb} MB")
    # model = torch.compile(model)
    # create the optimizer
    adam_optimizer = optimizer.AdamW(params = transformer_model.parameters(),
                                             lr = 1e-3,eps = 1e-8,
                                             weight_decay=0.01)
    t:int = 0
    logger.info(f"Beginning training loop, will warm up for {warmup_steps} warm up steps")
    # metrics_out = open(args.metrics_file,"a")
    # metrics_out.write("ITERATION\tWARMED_UP\tLEARNING_RATE\tTRAINING_LOSS\tFORWARD_TIME\tBACKWARD_TIME\tOPT_STEP\n")
    # metrics_out.flush()
    metrics_out = []
    for t in range(t,steps):
        metric = {}
        iteration_start = time.perf_counter()
        inputs, targets = get_batch(vocab_size = vocab_size, batch_size=batch_size,
                                                    context_length=context_length, device=device)
        forward_time = 0.
        backward_time = 0.
        opt_time = 0.
        loss_value = 0.
        #
        if training_mode=='forward':
            with torch.no_grad():
                logger.info("in forward iteration {t}")
                forward_start = get_time(device)
                model_output = transformer_model(inputs)
                forward_end = get_time(device)
                loss = cs336_basics.nn_utils.cross_entropy(inputs=model_output, targets=targets)
                forward_time = forward_end - forward_start
        else:
            logger.info("in training iteration {t}")
            adam_optimizer.zero_grad()  # Reset the gradients for all learnable parameters
            forward_start = get_time(device)
            model_output = transformer_model(inputs)
            forward_end = get_time(device)
            loss = cs336_basics.nn_utils.cross_entropy(inputs=model_output,targets =targets)
            forward_time = forward_end - forward_start
            backward_start = get_time(device)
            loss.backward()  # Run backward pass, which computes gradients
            backward_end = get_time(device)
            loss_value = loss.item()
            backward_time = backward_end - backward_start
            opt_start = get_time(device)
            adam_optimizer.step()
            opt_end = get_time(device)
            opt_time = opt_end - opt_start
        metric["iteration"] = t
        metric["warmed_up"] = t>=warmup_steps
        metric["loss"] = loss_value
        metric["forward_time"] = forward_time
        metric["backward_time"] = backward_time
        metric["opt_time"] = opt_time
        metrics_out.append(metric)
    # metrics_out.close()
    return metrics_out

@app.function(**config)
def underutilize(scale=1):
    records = []

    x = torch.randn(  # 🐌 2: not enough work to keep the GPU busy
        scale * 100, scale * 100, device="cuda"
    )

    for ii in range(10):
        x = x @ x

        class Record:  # 🐌 1: heavy Python work in the hot loop
            def __init__(self, value):
                self.value = value

        records.append(Record(ii))

    x[0][0].cpu()  # force a host sync for accurate timing


if __name__=="__main__":
    main()
