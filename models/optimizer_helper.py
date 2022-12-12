from torch import optim

def get_optim_and_scheduler(network, epochs, lr, nesterov=False, sched=None, hp_search_trial=None, opt='SGD'):
    
    params = network.parameters()
    
    if opt == 'SGD':
        optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)   
    elif opt == 'Adam':
        optimizer = optim.Adam(params, weight_decay=.0005, lr=lr)

    step_size = int(epochs * .8)
    if sched == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    elif sched == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=lr/10, last_epoch=-1)
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)
    return optimizer, scheduler
