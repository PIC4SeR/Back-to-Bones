import torch
import timm

from torchvision.models import resnet50, resnet18
from efficientnet_pytorch import EfficientNet

from models.vision_transformer import vit_base_patch16_224, deit_base_patch16_224
from models.convit import convit_base
    
    
def get_model(args):
    
    if args.meth == 'None':
        if args.network == 'resnet18':
            model = resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            model.fc = torch.nn.Linear(512, args.n_classes) # overwrite last fully connected layer
        elif args.network == 'resnet50':
            model = resnet50(pretrained=True) 
            model.fc = torch.nn.Linear(512*4, args.n_classes) # overwrite last fully connected layer
        elif args.network == 'efficientnetB0':
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.n_classes)
        elif args.network == 'efficientnetB1':
            model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=args.n_classes)
            args.image_size = 240
        elif args.network == 'efficientnetB2':
            model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=args.n_classes)
            args.image_size = 260
        elif args.network == 'efficientnetB3':
            model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=args.n_classes)
            args.image_size = 300
        elif args.network == 'vit_small16':
            model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=args.n_classes)
        elif args.network == 'vit_base16':
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.n_classes)
        elif args.network == 'vit_base32':
            model = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=args.n_classes)
        elif args.network == 'deit_base16':
            model = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=args.n_classes)
        elif args.network == 'convit_base':
            model = timm.create_model('convit_base', pretrained=True, num_classes=args.n_classes)
        elif args.network == 'convit_small':
            model = timm.create_model('convit_small', pretrained=True, num_classes=args.n_classes)
        elif args.network == 'levit_384':
            model = timm.create_model('levit_384', pretrained=True, num_classes=args.n_classes)
        else:
            raise Exception(f"Unknown model {args.network}")

    else:
        if args.network == 'vit_base16':
            model = vit_base_patch16_224(pretrained=True, num_classes=args.n_classes, meth=args.meth)
        elif args.network == 'convit_base':
            model = convit_base(pretrained=True, num_classes=args.n_classes, meth=args.meth)
        elif args.network == 'deit_base16':
            num_features = 768
            model = deit_base_patch16_224(pretrained=True, meth=args.meth)
            model.head = torch.nn.Linear(num_features, args.n_classes)
            model.num_classes = args.n_classes
        else:
            raise Exception(f"Unknown model {args.network}")

        
    return model, args