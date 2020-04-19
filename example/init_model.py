import torch

from torch_tresnet import tresnet_xl

# pretrianed on 224*224
model = tresnet_xl(pretrained=True, num_classes=10)

device = torch.device("cuda:2")
model = model.to(device)

inputs = torch.randn([32, 3, 224, 224]).to(device)
model(inputs)