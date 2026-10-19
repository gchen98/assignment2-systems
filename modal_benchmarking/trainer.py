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
        init_function: str = "initializer",
        function: str = "underutilize",
        label: Optional[str] = None,
        schedule=None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        print_rows: int = 10,
        kwargs_json_path: Optional[str] = None,
        model_params_file: Optional[str] = None,
):
    if kwargs_json_path is not None:  # use to pass arguments to function
        import json
        kwargs = json.loads(Path(kwargs_json_path).read_text())
    else:
        kwargs = {}


    import yaml

    # add things from the YAML file to kwargs
    with open(model_params_file, "r") as f_in:
        yaml_data = yaml.safe_load(f_in)
        model_params = yaml_data['model_params']
        kwargs['context_length'] = model_params['context_length']
        kwargs['embedding_dim'] = model_params['embedding_dim']
        kwargs['num_layers'] = model_params['num_layers']
        kwargs['num_heads'] = model_params['num_heads']
        kwargs['feed_forward_dim'] = model_params['feed_forward_dim']
        kwargs['rope_theta'] = model_params['rope_theta']

    results, remote_path = profile.remote(
        function,
        label=label,
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




@app.function(image = image, gpu=gpu)
def sanity_check(args):
    import torch
    print(torch.cuda.get_device_name("cuda"))
    print(torch.cuda.get_device_properties("cuda"))

@app.function(volumes={TRACE_DIR: traces}, **config)
def profile(
        function_name: str,
        label: Optional[str] = None,
        profiled_steps: int = 1,
        wait: int = 0,
        skip_first:int = 0,
        warmup_steps:int=1,
        schedule = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        print_rows: int = 0,
        **kwargs,
):
    # print(f"warmup steps: {warmup_steps} steps {steps}")
    from uuid import uuid4
    try:
        function = app.registered_functions[function_name]
    except KeyError:
        raise ValueError(f"Function {function_name} not found!")

    output_dir = (TRACE_DIR/(function_name+(f"_{label}" if label else "")) / str(uuid4()))
    output_dir.mkdir(parents = True, exist_ok= True)

    if schedule is None:
        # if steps < 3:
        #     raise ValueError("Steps must be at least 3 when using default schedule")
        total_steps = skip_first+wait+warmup_steps+profiled_steps
        # active_steps = total_steps - skip_first - wait - warmup_steps
        # total_steps = skip_first + wait + warmup_steps + steps
        schedule = {"skip_first":skip_first,"wait":wait, "warmup": warmup_steps, "active": profiled_steps, "repeat": 1 }
        print(f"Schedule {schedule}")
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
        import cs336_basics
        from cs336_basics import model,optimizer,nn_utils,data



        transformer_model = model.BasicsTransformerLM(vocab_size=kwargs['vocab_size'],
                                                      context_length=kwargs['context_length'],
                                                      d_model=kwargs['embedding_dim'],
                                                      num_layers=kwargs['num_layers'],
                                                      num_heads=kwargs['num_heads'],
                                                      d_ff=kwargs['feed_forward_dim'],
                                                      rope_theta=kwargs['rope_theta']).to(kwargs['device'])
        adam_optimizer = optimizer.AdamW(params=transformer_model.parameters(),
                                         lr=1e-3 , eps=1e-8,weight_decay=0.01)
        kwargs['transformer_model'] = transformer_model
        kwargs['adam_optimizer'] = adam_optimizer
        for a in range(total_steps):
            print(f"at step {a}")
            function.local(**kwargs)
            prof.step()
    if print_rows:
        print(
            prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=print_rows)
        )
    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key = lambda pth:pth.stat().st_mtime, reverse = True,
    )[0]

    print(f"trace saved to {trace_path.relative_to((TRACE_DIR))}")
    return trace_path.read_text(),trace_path.relative_to(TRACE_DIR)

@app.function()
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

# @app.function()
# def get_time(device: str) -> float:
#     if torch.cuda.is_available():
#         torch.cuda.synchronize(device)
#     return timeit.default_timer()

@app.function(**config)
def transformer_iteration(transformer_model,adam_optimizer,
                          vocab_size:int,batch_size:int,
                          context_length:int,device:str,training_mode:str,
                          **kwargs):
    import cs336_basics
    from cs336_basics import nn_utils
    inputs, targets = get_batch.local(vocab_size=vocab_size, batch_size=batch_size,
                                      context_length=context_length, device=device)
    if training_mode == 'forward':
        with torch.no_grad():
            # logger.info("in forward iteration {t}")
            # forward_start = get_time(device)
            model_output = transformer_model(inputs)
            # forward_end = get_time(device)
            loss = cs336_basics.nn_utils.cross_entropy(inputs=model_output, targets=targets)
            # forward_time = forward_end - forward_start
    elif training_mode == 'training':
        adam_optimizer.zero_grad()  # Reset the gradients for all learnable parameters
        # forward_start = get_time(device)
        model_output = transformer_model(inputs)
        # forward_end = get_time(device)
        loss = cs336_basics.nn_utils.cross_entropy(inputs=model_output, targets=targets)
        # forward_time = forward_end - forward_start
        # backward_start = get_time(device)
        loss.backward()  # Run backward pass, which computes gradients
        # backward_end = get_time(device)
        # loss_value = loss.item()
        # backward_time = backward_end - backward_start
        # opt_start = get_time(device)
        adam_optimizer.step()
        # opt_end = get_time(device)
        # opt_time = opt_end - opt_start



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
