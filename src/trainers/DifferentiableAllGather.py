import torch
import torch.distributed as dist

class DifferentiableAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        # Create a list for gathered tensors from all processes.
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        # Save local batch size for backward slicing.
        ctx.batch_size = tensor.shape[0]
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if not dist.is_initialized():
            return grad_output
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # Determine slice for the current process.
        start = rank * ctx.batch_size
        end = (rank + 1) * ctx.batch_size
        grad_input = grad_output[start:end].clone()
        # All-reduce gradients so that all processes get the same update.
        dist.all_reduce(grad_input)
        return grad_input
