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
         .add_local_python_source("cs336_basics"))

gpu="A100-80GB"
app = modal.App("model-benchmarking")
config = {"gpu":gpu, "image":image}

@app.local_entrypoint()
def main(

        function: str = "underutilize",
        label: Optional[str] = None,
        schedule=None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        timing:bool = False,
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


    (results,remote_path) = profile.remote(
        function,
        label=label,
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        timing=timing,
        print_rows=print_rows,
        **kwargs,
    )

    output_path = Path("/tmp") / remote_path.name
    if type(results)is str:
        output_path.write_text(results)
    elif type(results) is bytes:
        with open(output_path,"wb") as f:
            f.write(results)
    logger.info(f"trace saved locally at {output_path}")



@app.function(image = image, gpu=gpu)
def sanity_check(args):
    import torch
    logger.info(torch.cuda.get_device_name("cuda"))
    logger.info(torch.cuda.get_device_properties("cuda"))



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
        timing:bool = False,
        print_rows: int = 0,
        **kwargs,
):
    from contextlib import nullcontext
    # logger.debug(f"warmup steps: {warmup_steps} steps {steps}")
    from uuid import uuid4
    import numpy as np
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
        logger.info(f"Schedule {schedule}")
    schedule = torch.profiler.schedule(**schedule)
    # import cs336_basics
    from cs336_basics import model, optimizer, nn_utils, data
    transformer_model = model.BasicsTransformerLM(vocab_size=kwargs['vocab_size'],
                                                  context_length=kwargs['context_length'],
                                                  d_model=kwargs['embedding_dim'],
                                                  num_layers=kwargs['num_layers'],
                                                  num_heads=kwargs['num_heads'],
                                                  d_ff=kwargs['feed_forward_dim'],
                                                  rope_theta=kwargs['rope_theta']).to(kwargs['device'])
    adam_optimizer = optimizer.AdamW(params=transformer_model.parameters(),
                                     lr=1e-3, eps=1e-8, weight_decay=0.01)
    kwargs['transformer_model'] = transformer_model
    kwargs['adam_optimizer'] = adam_optimizer
    kwargs['logger'] = logger
    kwargs['timing'] = timing

    if timing:
        timing_dict = {}
        kwargs['timing_dict'] = timing_dict
        if profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        for a in range(total_steps):
            logger.info(f"timing step {a}")
            kwargs['is_warmup'] = (a < (total_steps - profiled_steps))
            function.local(**kwargs)
        out_path =  output_dir / "walltimes.txt"
        with open(out_path,"w") as f_out:
            for fn_name,times in timing_dict.items():
                np_times = np.array(times)
                f_out.write(f"{fn_name}\t{len(np_times)}\t{np.mean(np_times).item()}\t{np.std(np_times).item()}\n")
        if profile_memory:
            picklefile = output_dir / "memory_snapshot.pickle"
            torch.cuda.memory._dump_snapshot(str(picklefile))
            torch.cuda.memory._record_memory_history(enabled=None)
            with open(picklefile,"rb") as fin:
                raw_bytes = fin.read()
            return raw_bytes,picklefile.relative_to(TRACE_DIR)
        else:
            return out_path.read_text(),out_path.relative_to(TRACE_DIR)
    else:
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
            for a in range(total_steps):
                logger.info(f"at step {a}")
                function.local(**kwargs)
                prof.step()
            if print_rows:
                if profile_memory:
                    logger.info(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=print_rows))
                else:
                    logger.info(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=print_rows))
            trace_path = sorted(
                output_dir.glob("**/*.pt.trace.json"),
                key = lambda pth:pth.stat().st_mtime, reverse = True,
            )[0]

            logger.debug(f"trace saved to {trace_path.relative_to((TRACE_DIR))}")
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

@app.function()
def get_model_size(transformer_model)->float:
    model_size: int = 0
    for name, buf in transformer_model.named_buffers():
        num_elements = buf.numel()
        shape = buf.shape
        dtype = buf.dtype
        element_size = buf.element_size()
        logger.debug(f"buffer of {dtype} with shape {shape} and elements {num_elements} of size {element_size}")
        model_size += num_elements * element_size
    for param in transformer_model.parameters(recurse=True):
        num_elements = param.numel()
        shape = param.shape
        dtype = param.dtype
        element_size = param.element_size()
        logger.debug(f"param of {dtype} with shape {shape} and elements {num_elements} of size {element_size}")
        model_size += num_elements * element_size
    mb = model_size / (1024 ** 2)
    logger.info(f"Model size is {model_size} bytes, or {mb} MB")
    return mb

@app.function(**config)
def transformer_iteration(transformer_model,adam_optimizer,
                          vocab_size:int,batch_size:int,
                          context_length:int,device:str,training_mode:str,
                          precision_mode:str,dtype_str:str,
                          logger:logger,timing,
                          **kwargs):
    from contextlib import nullcontext,contextmanager
    import timeit
    import cs336_basics
    from cs336_basics import nn_utils
    inputs, targets = get_batch.local(vocab_size=vocab_size, batch_size=batch_size,
                                      context_length=context_length, device=device)

    dtype = torch.float32
    if precision_mode == "mixed":
        if dtype_str == "bf16":
            dtype = torch.bfloat16
        elif dtype_str == "fp16":
            dtype = torch.float16
        else:
            logger.warn(f"No valid dypes specified! Falling back to fixed precision")
            precision_mode = "fixed"
    autocast = precision_mode=="mixed"
    logger.info(f"Running on {dtype} in {precision_mode} mode autocast is {autocast}")
    run_backward = training_mode=='training'
    if timing:
        timing_dict: dict = kwargs['timing_dict']
        is_warmup: bool = kwargs['is_warmup']
    @contextmanager
    def cm_sync(fn_name):
        try:
            if timing:
                torch.cuda.synchronize(device)
                start = timeit.default_timer()
            yield
            if timing:
                torch.cuda.synchronize(device)
                end = timeit.default_timer()
                # logger.info(f"Elapsed time is {end-start}")
                if not is_warmup:
                    logger.info(f"Recording time for function {fn_name}")
                    timing_dict[fn_name] = timing_dict.get(fn_name,[])
                    timing_dict[fn_name].append((end-start))
        except:
            logger.error("Exception!")
        finally:
            logger.debug("Finally")


    cm_no_grad = torch.no_grad()  if training_mode=='forward' else nullcontext()
    cm_autocast = torch.autocast(device_type=device, dtype=dtype) if autocast else nullcontext()
    logger.debug(f'Training mode {training_mode}')
    with cm_no_grad:
        if run_backward:
            adam_optimizer.zero_grad()  # Reset the gradients for all learnable parameters
        with cm_autocast:
            with cm_sync("forward pass"):
                model_output = transformer_model(inputs)
            with cm_sync("loss"):
                loss = cs336_basics.nn_utils.cross_entropy(inputs=model_output, targets=targets)
            if run_backward:
                with cm_sync("backward pass"):
                    loss.backward()  # Run backward pass, which computes gradients
                with cm_sync("optimizer step"):
                    adam_optimizer.step()




if __name__=="__main__":
    main()
