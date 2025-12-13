import torchvision.models as models
import torch.nn as nn
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dist_model(model_cfg):
    """
    Build a distance regressor model based on the provided configuration.
    
    Args:
        model_cfg (dict): Configuration dictionary containing model parameters.
    
    Returns:
        nn.Module: The constructed distance regressor model.
    """
    input_channels = model_cfg.get('input_channels', 5)
    backbone = model_cfg.get('backbone', 'resnet50')

    # load model from path
    if 'model_path' in model_cfg:
        model_path = model_cfg['model_path']
        if not model_path.endswith('.pth'):
            raise ValueError("Model path must end with .pth")
        model = DistanceRegressor(input_channels=input_channels, backbone=backbone, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
    
    return model


class DistanceRegressor(nn.Module):
    """
    Currently used as follows:
    BACKBONE = 'convnext_small' 
    in_channels=6
    model = DistanceRegressor(input_channels=input_channels, backbone=BACKBONE, pretrained=True)
    """
    def __init__(self, input_channels=5, backbone='resnet50', pretrained=False):
        super().__init__()
        self.backbone_name = backbone
        
        # Load backbone
        if 'resnet' in backbone:
            self.model = getattr(models, backbone)(weights='IMAGENET1K_V1' if pretrained else None)
            
            # Replace first conv layer
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(input_channels, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding,
                                          bias=old_conv.bias is not None)
            
            # Initialize new channels
            if pretrained:
                with torch.no_grad():
                    self.model.conv1.weight[:, :3] = old_conv.weight
                    
                    if input_channels == 6:
                        # Index 3: Depth (Zero Init)
                        # Index 4,5: Masks (Kaiming Init)
                        nn.init.constant_(self.model.conv1.weight[:, 3], 0.0)
                        nn.init.kaiming_normal_(self.model.conv1.weight[:, 4:], mode='fan_out', nonlinearity='relu')
                    elif input_channels > 3:
                        # E.g. 5 channels (RGB + Masks), no Depth
                        nn.init.kaiming_normal_(self.model.conv1.weight[:, 3:], mode='fan_out', nonlinearity='relu')


            num_feats = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_feats, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        elif 'convnext' in backbone:
            self.model = getattr(models, backbone)(weights='IMAGENET1K_V1' if pretrained else None)
            
            # ConvNeXt first layer is features[0][0]
            old_conv = self.model.features[0][0]
            new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=old_conv.bias is not None)
            
            self.model.features[0][0] = new_conv
            
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    
                    if input_channels == 6:
                        # Index 3: Depth (Zero Init)
                        # Index 4,5: Masks (Kaiming Init)
                        nn.init.constant_(new_conv.weight[:, 3], 0.0)
                        nn.init.kaiming_normal_(new_conv.weight[:, 4:], mode='fan_out', nonlinearity='relu')
                    elif input_channels > 3:
                         nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out', nonlinearity='relu')
                    
                    if old_conv.bias is not None:
                        new_conv.bias = old_conv.bias

            # Replace classifier
            # ConvNeXt classifier: Sequential(LayerNorm2d, Flatten, Linear)
            # We need to find input features to the last Linear layer
            last_layer_idx = len(self.model.classifier) - 1
            num_feats = self.model.classifier[last_layer_idx].in_features
            
            self.model.classifier[last_layer_idx] = nn.Sequential(
                nn.Linear(num_feats, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        else:
            raise ValueError(f"Backbone {backbone} not supported yet.")

    def forward(self, x):
        return self.model(x).squeeze(1)
