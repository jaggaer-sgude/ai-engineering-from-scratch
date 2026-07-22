import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.backends.mps.is_available():
    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")

    torch.mps.synchronize()
    start = time.time()
    c_mps = a_mps @ b_mps
    torch.mps.synchronize()
    mps_time = time.time() - start
    print(f"MPS: {mps_time:.3f}s")
    # print(f"Speedup: {cpu_time / mps_time:.0f}x")