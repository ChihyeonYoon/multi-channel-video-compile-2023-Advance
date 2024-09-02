import torch.nn as nn
from torchvision.models import (
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_v2_b, Swin_V2_B_Weights,
    resnet18, ResNet18_Weights
)

def get_model(model_name='efficientnet_v2_m', num_classes=2, full_train=True):
    if model_name == 'efficientnet_v2_m':
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        model = efficientnet_v2_m(weights=weights)
        preprocess = weights.transforms()
        
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if not full_train:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        return model, preprocess
    
    elif model_name == 'efficientnet_v2_s':
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights)
        preprocess = weights.transforms()
        
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if not full_train:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        return model, preprocess

    elif model_name == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        preprocess = weights.transforms()
        
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        if not full_train:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        return model, preprocess
    
    elif model_name == 'mobilenet_v3_large':
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=weights)
        preprocess = weights.transforms()
        
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        if not full_train:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        return model, preprocess

    elif model_name == 'vit_b_16':
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        preprocess = weights.transforms()
        
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        if not full_train:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.heads.head.parameters():
                param.requires_grad = True
        return model, preprocess

    elif model_name == 'swin_v2_b':
        weights = Swin_V2_B_Weights.IMAGENET1K_V1
        model = swin_v2_b(weights=weights)
        preprocess = weights.transforms()
        
        model.head = nn.Linear(model.head.in_features, num_classes)
        if not full_train:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        return model, preprocess

    elif model_name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        preprocess = weights.transforms()
        
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if not full_train:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model, preprocess

# models
# ['efficientnet_v2_s', 'mobilenet_v3_small', mobilenet_v3_large, 'vit_b_16', 'swin_v2_b', 'resnet18']

