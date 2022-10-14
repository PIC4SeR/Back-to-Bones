from torch import optim


def get_optim_and_scheduler(network, epochs, lr, train_all, nesterov=False, sched=None, hp_search_trial=None, opt='SGD'):
    
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
        
    if opt == 'SGD':
        optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)   
    elif opt == 'Adam':
        optimizer = optim.Adam(params, weight_decay=.0005, lr=lr)

    if sched == 'step':
        step_size = int(epochs * .8)
        # print("Step size: %d" % step_size)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    elif sched == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=lr/10, last_epoch=-1)
    elif sched == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)
    else:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)
        
    return optimizer, scheduler
