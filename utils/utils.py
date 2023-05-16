import argparse
from time import time
from os.path import join, dirname


# high level wrapper for tf_logger.TFLogger
class Logger():
    def __init__(self, args, update_frequency=10):
        self.current_epoch = 0
        self.max_epochs = args.epochs
        self.last_update = time()
        self.start_time = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency
        folder, logname = self.get_name_from_args(args)
        log_path = join(args.log_path, folder, logname)
        self.current_iter = 0

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update = time()
        self.lrs = learning_rates
        #print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs]))
        self._clean_epoch_stats()

    def log(self, it, iters, losses, samples_right, total_samples):
        self.current_iter += 1
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses.items()])
        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
        self.total += total_samples
        acc_string = ", ".join(["%s : %.2f" % (k, 100 * (v / total_samples)) for k, v in samples_right.items()])
        if it % self.update_f == 0:
            print("%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (it, iters, self.current_epoch, self.max_epochs, loss_string,
                                                                acc_string, total_samples))
         
    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.total = 0

    def log_test(self, phase, accuracies):
        print("Accuracies on %s: " % phase + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()]))
   
    def save_best(self, val_test, best_test):
        print("It took %g" % (time() - self.start_time))
       
    @staticmethod
    def get_name_from_args(args):
        folder_name = "%s_to_%s" % ("-".join(sorted(args.source)), args.target)
        name = "eps%d_bs%d_lr%g_class%d" % (args.epochs, args.batch_size, args.lr, args.n_classes)
        name += "_%d" % int(time() % 1000)
        return folder_name, name

    
class LoadFromFile(argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

            
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Utils
    parser.add_argument('--config', type=open, action=LoadFromFile, default='utils/config.txt')
    parser.add_argument("--cuda", default=0, type=int, help="Select cuda device")
    parser.add_argument("--num_workers", default=4, type=int, help="Select number of workers")
    parser.add_argument("--log_path", default='logs/', help="log directory")
    parser.add_argument("--verbose", default=False, type=bool, help="verbose")
    parser.add_argument("--training", default=True, type=bool, help="train")
    
    # Dataset
    parser.add_argument("--dg", default=1, type=bool, help="whether to train in DG mode")
    parser.add_argument("--data_path", default='Datasets/', help="dataset folder")
    parser.add_argument("--dataset", default='PACS', help="dataset name")
    parser.add_argument("--source", help="Source", nargs='+')
    parser.add_argument("--target", help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")

    # Data Augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")

    # Model and Training
    parser.add_argument("--network", help="Which network to use", default="vit_base16")
    parser.add_argument("--lr", "-l", type=float, default=.00001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr_scheduler", default='cos', help="Select LR scheduler") 
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--load_model", default=False, type=bool, help="Load trained model weights") 
    parser.add_argument("--optimizer", default='Adam', help="Select optimizer")
    
    # DG Methodology
    parser.add_argument("--meth", default=None)
    
    return parser.parse_args()