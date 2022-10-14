import torch
import optuna
from optuna.trial import TrialState
import numpy as np
import joblib
import argparse

from data.data_helper import available_datasets
from train import Trainer

def get_default_args():
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
    
    return parser.parse_args()


def get_args(trial):
    args = get_default_args()
    
    if args.dataset == 'PACS':
        args.n_classes = 7
        if args.target_domain == 'photo':
            args.source = ['art_painting', 'cartoon', 'sketch']
            args.target = 'photo'
        elif args.target_domain == 'sketch':
            args.source = ['art_painting', 'cartoon', 'photo']
            args.target = 'sketch'
        elif args.target_domain == 'cartoon':
            args.source = ['art_painting', 'photo', 'sketch']
            args.target = 'cartoon'
        else:
            args.source = ['photo', 'cartoon', 'sketch']
            args.target = 'art_painting'
 
    if args.dataset == 'VLCS':
        args.n_classes = 5
        if args.target_domain == "CALTECH":
            args.source = ["LABELME", "PASCAL", "SUN"]
            args.target = 'CALTECH'
        elif args.target_domain == "LABELME":
            args.source = ["CALTECH", "PASCAL", "SUN"]
            args.target = "LABELME"
        elif args.target_domain == 'PASCAL':
            args.source = ["LABELME", "CALTECH", "SUN"]
            args.target = 'PASCAL'
        else:
            args.source = ["LABELME", "PASCAL", "CALTECH"]
            args.target = 'SUN'

    if args.dataset == 'OfficeHome':
        args.n_classes = 65
        if args.target_domain == 'Real_World':
            args.source = ['Art', 'Clipart', 'Product']
            args.target = 'Real_World'
        elif args.target_domain == 'Art':
            args.source = ['Clipart', 'Real_World', 'Product']
            args.target = 'Art'
        elif args.target_domain == 'Clipart':
            args.source = ['Art', 'Product', 'Real_World']
            args.target = 'Clipart'
        else:
            args.source = ['Art', 'Clipart', 'Real_World']
            args.target = 'Product'

    if args.dataset == 'TerraIncognita':
        args.n_classes = 16
        if args.target_domain == '100':
            args.source = ['38', '43', '46']
            args.target = '100'
        elif args.target_domain == '38':
            args.source = ['43', '46', '100']
            args.target = '38'
        elif args.target_domain == '43':
            args.source = ['46', '100', '38']
            args.target = '43'
        else:
            args.source = ['100', '38', '43']
            args.target = '46'
            
    if args.dataset == 'ColoredMNIST':
        args.n_classes = 2
        if args.target_domain == 'train_1':
            args.source = ['train_2', 'test']
            args.target = 'train_1'
        if args.target_domain == 'train_2':
            args.source = ['train_1', 'test']
            args.target = 'train_2'
        else:
            args.source = ['train_2', 'train_1']
            args.target = 'test'
            
    # set search parameters
    args.learning_rate = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    args.lr_scheduler = trial.suggest_categorical("lr_scheduler", ['const'])
    args.batch_size = trial.suggest_categorical("batch_size", [32])
    args.optimizer = trial.suggest_categorical("optimizer", ["SGD"])
    return args
    
def objective(trial):

    args = get_args(trial)
    device = torch.cuda.set_device(args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = Trainer(args, device, trial)
    print(f"Training: model={args.network}, bs={args.batch_size}, lr={args.learning_rate}, scheduler={args.lr_scheduler}, epochs={args.epochs}, cuda={args.cuda}, target={args.target_domain}")
    _, _, accuracy, val_accuracy = trainer.do_training()
    
    return val_accuracy


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    
    search_space = {"lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 
                    "lr_scheduler":["const"], 
                    "batch_size":[32], 
                    "optimizer": ["SGD"]}
    
    study = optuna.create_study(direction="maximize", 
                                pruner=optuna.pruners.MedianPruner(),            
                                sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=5)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    joblib.dump(study, f"study_mnist.pkl")