import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=1000, pretrained=False):
    model = models.vgg16(pretrained)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    return model

def get_update_param_names(mode=''):
    update_param = []
    update_param.append({'all': False, 'names': ["classifier.6.weight", "classifier.6.bias"]})
    update_param.append({'all': False, 'names': ["classifier.0.weight", "classifier.0.bias","classifier.3.weight", "classifier.3.bias"]})
    update_param.append({'all': True, 'names': ["features"]})
    update_params_finetune = [ update_param[0], update_param[1], update_param[2] ]
    update_params_transfer = [ update_param[0] ]
    update_params_full = [ {'all': True, 'names': ["classifier", "feature"]} ]
    update_params = {'full': update_params_full, 'transfer': update_params_transfer, 'finetune': update_params_finetune }

    return update_params[mode]
