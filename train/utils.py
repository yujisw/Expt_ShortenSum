import torch

def print_gpu_status():
    cuda_count = torch.cuda.device_count()
    for i in range(cuda_count):
        print(torch.cuda.get_device_properties(i))
        print("Reserved:", torch.cuda.memory_reserved(i)/1000000, "MiB", "Allocated:", torch.cuda.memory_allocated(i)/1000000, "MiB", "Reserved - Allocated:", (torch.cuda.memory_reserved(i)-torch.cuda.memory_allocated(i))/1000000, "MiB")
        # print("Reserved:", torch.cuda.memory_reserved(i)/1000000, "MiB")
        # print("Allocated:", torch.cuda.memory_allocated(i)/1000000, "MiB")
        # print("Reserved - Allocated:", (torch.cuda.memory_reserved(i)-torch.cuda.memory_allocated(i))/1000000, "MiB")

