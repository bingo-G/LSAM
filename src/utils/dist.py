"""
Distributed training utilities.
Adapted from legacy project patterns - standalone implementation.
"""

import os
import torch
import torch.distributed as dist
import builtins


def init_distributed():
    """Initialize DDP from environment variables set by torchrun.

    No-arg convenience wrapper: reads RANK, WORLD_SIZE, LOCAL_RANK from env.
    Safe to call even when not launched via torchrun (falls back to single-GPU).
    """
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print('Not using distributed mode (RANK/WORLD_SIZE not set)', flush=True)
        return

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    has_cuda = torch.cuda.is_available()
    backend = os.environ.get('HMF_VQA_DIST_BACKEND', '')
    backend = backend.strip().lower() if backend else ('nccl' if has_cuda else 'gloo')
    if backend == 'nccl' and not has_cuda:
        backend = 'gloo'

    if has_cuda:
        torch.cuda.set_device(local_rank)

    print(
        f'| distributed init (rank {rank}, local_rank {local_rank}, world {world_size}, backend={backend})',
        flush=True,
    )
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()
    setup_for_distributed(rank == 0)


def init_distributed_mode(args):
    """Initialize DDP from environment variables (torchrun / SLURM). Legacy API."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ.get('SLURM_NTASKS', 1))
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        return

    args.distributed = True
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.set_device(args.local_rank)

    backend = str(getattr(args, 'dist_backend', 'nccl')).lower()
    if backend == 'nccl' and not has_cuda:
        backend = 'gloo'
    dist_url = getattr(args, 'dist_url', 'env://')

    print(f'| distributed init (rank {args.rank}, world {args.world_size}): {dist_url}', flush=True)
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master: bool):
    """Disable printing and suppress logging on non-master processes."""
    import logging as _logging
    import warnings as _warnings
    builtin_print = builtins.print

    def _print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = _print

    # Suppress console logging and warnings on non-master ranks
    if not is_master:
        _logging.basicConfig(level=_logging.WARNING)
        for name in list(_logging.Logger.manager.loggerDict.keys()):
            _logging.getLogger(name).setLevel(_logging.WARNING)
        _logging.getLogger().setLevel(_logging.WARNING)
        _warnings.filterwarnings('ignore')


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and divide by world size."""
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> list:
    """Gather tensors from all ranks."""
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


def all_gather_objects(obj):
    """Gather arbitrary objects from all ranks."""
    world_size = get_world_size()
    if world_size < 2:
        return [obj]
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, obj)
    return output
