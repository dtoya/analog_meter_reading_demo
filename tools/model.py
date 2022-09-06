
def get_optim_params(model, update_param_names, lr):
    params_to_update = [[] for _ in range(len(update_param_names))]
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i, param_names in enumerate(update_param_names):
            if param_names['all']:
                for update_name in param_names['names']:
                    if update_name in name:
                        param.requires_grad = True
                        params_to_update[i].append(param)
                        print('params_to_update[{}]: {}'.format(i, name))
            else:
                if name in param_names['names']:
                    param.requires_grad = True
                    params_to_update[i].append(param)
                    print('params_to_update[{}]: {}'.format(i, name))

    print('lr = {}'.format(lr))
    optim_params = []
    for i, param_names in enumerate(update_param_names):
        optim_param = {'params': params_to_update[i], 'lr': lr[i]}
        optim_params.append(optim_param)
    
    return optim_params
