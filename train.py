import argparse
import optuna
import torch
from torch import nn
from data import data_helper
from data.data_helper import available_datasets
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
from torchvision.models import resnet18, resnet50
from efficientnet_pytorch import EfficientNet
import timm

from utils.excellogger import Logger as ExcelLogger
from utils.excellogger import Column
from utils.dino_deit import VitGenerator
from utils.mnist_cnn import *


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--image_size", type=int, default=28, help="Image size")

    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--n_classes", "-c", type=int, default=65, help="Number of classes: PACS(7), VLCS(5), OfficeHome(65)")
    parser.add_argument("--tf_logger", type=bool, default=False, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")

    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--cuda", default=0, type=int, help="Select cuda device")
    parser.add_argument("--load_model", default=False, type=bool, help="Load trained model weights") 
    
    # important parameters
    parser.add_argument("--dataset", default='ColoredMNIST', help="dataset name")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--network", help="Which network to use", default="mnist_cnn")
    parser.add_argument("--target_domain", default='test', help="Select target PACS domain")
    
    # random search parameters
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--lr_scheduler", default='step', help="Select LR scheduler") 
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--optimizer", default='SGD', help="Select optimizer")
    parser.add_argument("--pretrained", default=False, type=bool, help="Load pretrained model weights") 
    parser.add_argument("--batch_multiplier", type=int, default=1)
    parser.add_argument("--trial", type=int, default=0)
    
    return parser.parse_args()

class Trainer:
    def __init__(self, args, device, trial=None):
        self.args = args
        self.device = device
        self.trial = trial
        self.excellogger = ExcelLogger('19SarFsJSjcW6Fbc18VRtDnwPDUczjJwGPhTXOIqdl5g')
        
        if args.network == 'mnist_cnn':
            model = MNIST_CNN(input_shape=(3,28,28))
            self.model = model.to(self.device)
        elif args.network == 'resnet18':
            model = resnet18(pretrained=args.pretrained)
            model.fc = nn.Linear(512, args.n_classes) # overwrite last fully connected layer
            self.model = model.to(self.device)
        elif args.network == 'resnet50':
            model = resnet50(pretrained=args.pretrained) 
            model.fc = nn.Linear(512*4, args.n_classes) # overwrite last fully connected layer
            self.model = model.to(self.device)
        elif args.network == 'resnet50A1':
            model = timm.create_model('resnet50', pretrained=args.pretrained, num_classes=args.n_classes)
            model.fc = nn.Linear(512*4, args.n_classes) # overwrite last fully connected layer
            self.model = model.to(self.device)
        elif args.network == 'efficientnetB0':
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.n_classes)
            self.model = model.to(self.device)
        elif args.network == 'efficientnetB1':
            model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=args.n_classes)
            self.model = nn.DataParallel(model, device_ids=[1], output_device=self.device)
            args.image_size = 240
        elif args.network == 'efficientnetB2':
            model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=args.n_classes).cuda(self.device)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device)
            args.image_size = 260
        elif args.network == 'efficientnetB3':
            model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=args.n_classes).cuda(self.device)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device)
            args.image_size = 300
        elif args.network == 'vit_small16':
            model = timm.create_model('vit_small_patch16_224', pretrained=args.pretrained, num_classes=args.n_classes).cuda(self.device)
            self.model = model.to(self.device)
        elif args.network == 'vit_base16':
            model = timm.create_model('vit_base_patch16_224', pretrained=args.pretrained, num_classes=args.n_classes).cuda(self.device)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device)
        elif args.network == 'vit_base32':
            model = timm.create_model('vit_base_patch32_224', pretrained=args.pretrained, num_classes=args.n_classes).cuda(self.device)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device)
        elif args.network == 'deit_small16':
            model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=args.n_classes).cuda(self.device)
            w = torch.load('./deit_small_patch16_224-cd65a155.pth')['model']
            del w['head.weight'], w['head.bias']
            model.load_state_dict(w, strict=False)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device) 
        elif args.network == 'deit_base16':
            model = timm.create_model('deit_base_patch16_224', pretrained=args.pretrained, num_classes=args.n_classes).cuda(self.device)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device) 
        elif args.network == 'convit_base':
            model = timm.create_model('convit_base', pretrained=args.pretrained, num_classes=args.n_classes).cuda(self.device)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device)
        elif args.network == 'convit_small':
            model = timm.create_model('convit_small', pretrained=args.pretrained, num_classes=args.n_classes).cuda(self.device)
            self.model = model.to(self.device)
        elif args.network == 'levit_384':
            model = timm.create_model('levit_384', pretrained=args.pretrained, num_classes=args.n_classes).cuda(self.device)
            self.model = model.to(self.device)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device)
        elif args.network == 'dino_base_16':
            gen = VitGenerator('vit_base', patch_size=16, num_classes=args.n_classes, evaluate=False, random= not args.pretrained, verbose=True)
            model = gen().to(self.device)
            self.model = nn.DataParallel(model, device_ids=[0,1], output_device=self.device)
            
        # print(self.model)
        self.save_model_path = './bin/best_' + args.network + "_" + args.target_domain + '.pth'
        if args.load_model:
            self.model.load_state_dict(torch.load(self.save_model_path))
        
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_test_dataloader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(network=model, epochs=args.epochs, 
                                                                 lr=args.learning_rate, train_all=args.train_all,
                                                                 nesterov=args.nesterov, sched=args.lr_scheduler, 
                                                                 opt=args.optimizer)
        self.n_classes = args.n_classes
     
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)
        self.model.train()
        
        count = 0
        
        for it, ((data, class_l), d_idx) in enumerate(self.source_loader):
            data, class_l, d_idx = data.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            
            #### SuperBatch
            
            if count == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                count = self.args.batch_multiplier

            class_logit = self.model(data)
            loss = criterion(class_logit, class_l) / self.args.batch_multiplier
            _, cls_pred = class_logit.max(dim=1)
            
            loss.backward()
            count -= 1
            
            
            
#             self.optimizer.zero_grad()

#             class_logit = self.model(data)
#             loss = criterion(class_logit, class_l)
#             _, cls_pred = class_logit.max(dim=1)

#             loss.backward()
#             self.optimizer.step()
#             del loss, class_logit

            self.logger.log(it, len(self.source_loader),
            {"class": loss.item()},
            {"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)

                class_correct = self.do_test(loader)

                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
                if self.trial is not None:
                    self.trial.report(class_acc, epoch)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            # save best model
            if class_acc > self.best_acc:
                self.best_acc = class_acc
                torch.save(self.model.state_dict(), self.save_model_path)
                

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, class_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct


    def do_training(self):
        self.logger = Logger(self.args, update_frequency=1000)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        self.best_acc = 0
        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_last_lr())
            self._do_epoch(self.current_epoch)
            self.scheduler.step()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best_val = val_res.argmax()
        idx_best_test = test_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best_val], test_res.max(), idx_best_val))
        
        data = [test_res[idx_best_val].item()]
        #self.excellogger.write(data, 'RSB', self.args.rowExcel, self.args.columnExcel)
        self.logger.save_best(test_res[idx_best_val], test_res.max())
        
        return self.logger, self.model, test_res[idx_best_val], val_res.max()


def main():
    args = get_args()
    if args.dataset == 'PACS':
        args.n_classes = 7
        args.rowExcel = 2
        if args.target_domain == 'photo':
            args.columnExcel = 'B'
            args.source = ['art_painting', 'cartoon', 'sketch']
            args.target = 'photo'
        elif args.target_domain == 'art_painting':
            args.columnExcel = 'G'
            args.source = ['photo', 'cartoon', 'sketch']
            args.target = 'art_painting'
        elif args.target_domain == 'cartoon':
            args.columnExcel = 'L'
            args.source = ['art_painting', 'photo', 'sketch']
            args.target = 'cartoon'
        else:
            args.columnExcel = 'Q'
            args.source = ['art_painting', 'cartoon', 'photo']
            args.target = 'sketch'
 
    if args.dataset == 'VLCS':
        args.rowExcel = 3
        args.n_classes = 5
        if args.target_domain == "CALTECH":
            args.columnExcel = 'B'
            args.source = ["LABELME", "PASCAL", "SUN"]
            args.target = 'CALTECH'
        elif args.target_domain == "LABELME":
            args.columnExcel = 'G'
            args.source = ["CALTECH", "PASCAL", "SUN"]
            args.target = "LABELME"
        elif args.target_domain == 'PASCAL':
            args.columnExcel = 'L'
            args.source = ["LABELME", "CALTECH", "SUN"]
            args.target = 'PASCAL'
        else:
            args.source = ["LABELME", "PASCAL", "CALTECH"]
            args.columnExcel = 'Q'
            args.target = 'SUN'

    if args.dataset == 'OfficeHome':
        args.rowExcel = 4
        args.n_classes = 65
        if args.target_domain == 'Product':
            args.source = ['Art', 'Clipart', 'Real_World']
            args.columnExcel = 'B'
            args.target = 'Product'
        elif args.target_domain == 'Art':
            args.columnExcel = 'G'
            args.source = ['Clipart', 'Real_World', 'Product']
            args.target = 'Art'
        elif args.target_domain == 'Clipart':
            args.columnExcel = 'L'
            args.source = ['Art', 'Product', 'Real_World']
            args.target = 'Clipart'
        else:
            args.columnExcel = 'Q'
            args.source = ['Art', 'Clipart', 'Product']
            args.target = 'Real_World'

    if args.dataset == 'TerraIncognita':
        args.rowExcel = 5
        args.n_classes = 16
        if args.target_domain == '100':
            args.columnExcel = 'B'
            args.source = ['38', '43', '46']
            args.target = '100'
        elif args.target_domain == '38':
            args.columnExcel = 'G'
            args.source = ['43', '46', '100']
            args.target = '38'
        elif args.target_domain == '43':
            args.columnExcel = 'L'
            args.source = ['46', '100', '38']
            args.target = '43'
        else:
            args.columnExcel = 'Q'
            args.source = ['100', '38', '43']
            args.target = '46'
    
    if args.dataset == 'ColoredMNIST':
        args.rowExcel = 6
        args.n_classes = 2
        args.columnExcel = 'AAA'
        if args.target_domain == 'train_1':
            args.source = ['train_2', 'test']
            args.target = 'train_1'
        if args.target_domain == 'train_2':
            args.source = ['train_1', 'test']
            args.target = 'train_2'
        else:
            args.source = ['train_2', 'train_1']
            args.target = 'test'
    
    args.columnExcel = Column(args.columnExcel)
    #print(args.columnExcel, args.trial)
    args.columnExcel = args.columnExcel + args.trial
            
    # --------------------------------------------
    print("Target domain: {}".format(args.target))
    device = torch.cuda.set_device(args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    
    print(f"Training: model={args.network}, bs={args.batch_size}, lr={args.learning_rate}, scheduler={args.lr_scheduler}, epochs={args.epochs}, cuda={args.cuda}, target={args.target_domain}")
    
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
