import argparse

import numpy as np
import optuna
import torch

from data import data_helper
from models.optimizer_helper import get_optim_and_scheduler
from utils.utils import Logger, get_args

from models.utils import get_model


class Trainer:
    def __init__(self, args, device, trial=None):
        self.args = args
        self.device = device
        self.trial = trial
            
        self.model, self.args = get_model(self.args)
        self.model = torch.nn.DataParallel(self.model, device_ids=[self.args.cuda], output_device=self.device)
     
        self.save_model_path = './bin/best_' + self.args.network + "_" + self.args.target + '.pth'
        
        if self.args.load_model:
            self.model.load_state_dict(torch.load(self.save_model_path))
        
        self.args.batch_size = self.args.batch_size // 3 
        self.source_loader, self.val_loader = data_helper.get_dataset_list(self.args)
        self.len_dataloader = -1

        self.target_loader = data_helper.get_test_dataloader(self.args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}

        if self.args.verbose:
            print("Dataset size: train %d, val %d, test %d" % (
            len(self.val_loader.dataset)*9, len(self.val_loader.dataset),
            len(self.target_loader.dataset)))
            
        self.optimizer, self.scheduler = get_optim_and_scheduler(network=self.model, epochs=self.args.epochs, 
                                                                 lr=self.args.lr, nesterov=self.args.nesterov, 
                                                                 sched=self.args.lr_scheduler)      
        if self.args.meth == 'None':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.criterion = self.criterion.to(self.device)

    def _do_epoch(self, epoch=None):
        self.model.train()
        
        for _ in range(len(self.val_loader.dataset)*3//self.args.batch_size):
            minibatches_device = [next(iter(loader)) for loader in self.source_loader]
            
            if self.args.meth == 'None':
                all_x = torch.cat([x for x, y in minibatches_device]).cuda()
                all_y = torch.cat([y for x, y in minibatches_device]).cuda()
                pred_y = self.model(all_x)
                loss = self.criterion(pred_y, all_y)
                
            else:
                loss = self.model(minibatches_device)
            
            self.optimizer.zero_grad()
            loss.backward(loss)
            self.optimizer.step()
        
            del loss
            
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
                        
            # Save best model
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
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        self.best_acc = 0
        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_last_lr())
            self._do_epoch(self.current_epoch)
            self.scheduler.step()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        
        if self.args.verbose:
            print("Best val %g, Corresponding test %g - Best test: %g, Best epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        else:
            print("Best val %g, Corresponding test %g" % (val_res.max(), test_res[idx_best]))
        
        self.logger.save_best(test_res[idx_best], test_res.max())
        
        return self.logger, self.model, test_res[idx_best]

    
def main():
    args = get_args()
    print(args)
    args = data_helper.get_source_domains(args)
    
    device = torch.cuda.set_device(args.cuda)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(args, device)
    
    if args.verbose:
        print(f'''Training: model={args.network}, bs={args.batch_size}, lr={args.lr}, scheduler={args.lr_scheduler},
          epochs={args.epochs}, cuda={args.cuda}, target={args.target}, meth={args.meth}''')
    
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()