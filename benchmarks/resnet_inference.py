import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.conv(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def benchmark_resnet(runs=10):
    model = SimpleResNet()
    input_tensor = torch.randn(1, 3, 224, 224)
    times = []
    for _ in range(runs):
        start = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        times.append(time.time() - start)
    mean = np.mean(times)
    std = np.std(times)
    print(f"ResNet-50 Inference: Mean {mean:.4f}s/image ± {std:.4f}s")
    return {'mean': mean, 'std': std}

if __name__ == "__main__":
    import numpy as np
    benchmark_resnet()
