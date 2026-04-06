import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA có sẵn? : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"Khả năng tính toán: {torch.cuda.get_device_capability(0)}")
else:
    print("Vẫn đang dùng CPU")