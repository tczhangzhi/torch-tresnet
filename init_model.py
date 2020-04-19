from torch_tresnet import tresnet_xl

# pretrianed on 224*224
model = tresnet_xl(pretrained=True, num_classes=10)