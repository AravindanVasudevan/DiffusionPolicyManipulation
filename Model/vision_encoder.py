import torch.nn as nn
import torchvision



def build_resnet18_encoder(pretrained: bool = True) -> nn.Module:

    weights = "IMAGENET1K_V1" if pretrained else None
    net = torchvision.models.resnet18(weights=weights)

    def _replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.GroupNorm(min(16, child.num_features), child.num_features))
            else:
                _replace(child)

    _replace(net)
    net.fc = nn.Identity()

    return net