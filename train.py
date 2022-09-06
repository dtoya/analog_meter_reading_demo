import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
#
import pandas as pd
from tqdm import tqdm
import sys,os
import numpy as np
import cv2
# Private
import models
import tools

# For repeatability of result
torch.manual_seed(1234)
#np.random.seed(1234)
#random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = tools.cargparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--config_file', default='./config.json', help='Label added output output log/pth file name.')
parser.add_argument('--epochs', default=10, help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, help='Batch size')
parser.add_argument('--image_size', default=224, help='Image size [h, w] or scalar')
parser.add_argument('--workers', default=4, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs_save', default=-1, help='number of epochs to save pth file')
parser.add_argument('--result_path', default='./result', help='number of epochs to save pth file')
parser.add_argument('--use_tqdm', default=False, action='store_true', help='Use tqdm module to show progress.')
parser.add_argument('--use_tensorboard', default=False, action='store_true', help='Use Tensorboard to monitor status')
parser.add_argument('--file_name_label', default='', help='Label added output output log/pth file name.')
parser.add_argument('--device', default='cuda:0', help='Device for computation')
parser.add_argument('--training_mode', default='full', help='Mode for training (full/transfer/finetune).')
parser.add_argument('--learning_rate', default=[1e-3, 5e-4, 1e-4], nargs='*', help='Learning Rate')
parser.add_argument('--criteria', default=0.05, help='Criteria on difference between output and label to pass')
parser.add_argument('--loss_function', default='MSELoss', help='Select Loss function')
parser.add_argument('--optimizer', default='Adam', help='Select optimizer')
parser.add_argument('--data_root', default='./data', help='Select optimizer')
parser.add_argument('--save_onnx', default=False, action='store_true', help='Save model in onnx format')
parser.add_argument('--use_pytorch_pretrained', default=False, action='store_true', help='Use pretrained weights for Pytorch models.')
parser.add_argument('--load_weights', default='', help='Load weights in checkpoint file')
parser.add_argument('--model_name', default='alexnet', help='Model name')
config = parser.parse_args()

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                #transforms.RandomResizedCrop(
                #   resize, scale=(0.5, 1.0)),
                transforms.Resize(resize),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                #transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):

        return self.data_transform[phase](img)

def run_epoch(phase, model, dataloaders, optimizer, criterion, device):
    epoch_loss = 0.0
    epoch_correct = 0.0
    
    if phase == 'train':
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    if config.use_tqdm:
        dataloader = tqdm(dataloaders[phase])
    else:
        dataloader = dataloaders[phase]

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        labels = torch.unsqueeze(labels,1)
        loss = criterion(outputs, labels)

        if phase == 'train':
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        epoch_loss += loss.item() * inputs.size(0)
        diff = torch.abs(outputs - labels)
        corrects = list(torch.where(diff < config.criteria))[0]
        epoch_correct += corrects.size(0)

    datasize = len(dataloaders[phase].dataset)
    epoch_loss = epoch_loss / datasize
    epoch_acc = epoch_correct / datasize
    return epoch_loss, epoch_acc


def test(model, dataloaders, device):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():

        if config.use_tqdm:
            dataloader = tqdm(dataloaders['val'])
        else:
            dataloader = dataloaders['val']
     
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(images.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def imcheck(img):
    img = torchvision.utils.make_grid(img)
    img = (img / 2 + 0.5)*255  # unnormalize
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0)) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(config.result_path+'/transformed_image.jpg',img)
    
def main():
    
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if config.training_mode == 'full':
        if config.use_pytorch_pretrained:
            print('WARNING: Do not use Pytorch pretrained model in full training. Changed training mode or config.use_pytorch_pretrained.')
            config.use_pytorch_pretrained = False
    else:
        if (len(config.load_weights) == 0) and (not config.use_pytorch_pretrained):
            print('WARNING: No pretrained weight is available. Use Pytorch pretrained model. Changed config.use_pytorch_pretrained.')
            config.use_pytorch_pretrained = True
    print('Config Parameters:')
    parser.dump(config, sys.stdout, indent=2)
    print('')
    with open('{}/config.json'.format(config.result_path), 'w') as f:
        parser.dump(config, f, indent=2)

    if config.use_tensorboard:
        writer = SummaryWriter(log_dir="./logs")

    if torch.cuda.is_available():
        device = torch.device(config.device)
    else:
        if config.device == 'cuda:0':
            print('Warning: CUDA deivce is not available. Use CPU.')
        device = torch.device('cpu')

    # Configure Image Transform
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = ImageTransform(config.image_size, mean, std)
     
    # Configure dataset (Use test dataset for validation.)    
    dataset = tools.meterdata.MeterDataset
    dataset_args = {'root': config.data_root, 'transform': transform} 
    dataloaders = tools.dataloader.get_dataloaders(dataset, dataset_args, config.batch_size, config.workers)

    # Configure network
    if config.model_name == 'alexnet':
        model_name = models.alexnet
    model = model_name.get_model(num_classes=1, pretrained=config.use_pytorch_pretrained)
    if len(config.load_weights) != 0:
        model.load_state_dict(torch.load(config.load_weights))
    optim_params = tools.model.get_optim_params(model, model_name.get_update_param_names(config.training_mode), lr=config.learning_rate)
    model.to(device)
  
    inputs, labels = next(iter(dataloaders['train']))
    imcheck(inputs[0])
    shape = tuple(inputs[0].size())
    summary(model, shape)

    if config.loss_function == 'L1Loss':
        criterion = nn.L1Loss()
    else: 
        criterion = nn.MSELoss()

    if config.optimizer == 'SGD':
        optimizer = optim.SGD(optim_params, momentum=0.9)
    else:
        optimizer = optim.Adam(optim_params)

    for epoch in range(config.epochs+1):  # loop over the dataset multiple times

        if epoch != 0:
            train_loss, train_acc = run_epoch('train', model, dataloaders, optimizer, criterion, device)

        val_loss, val_acc = run_epoch('val', model, dataloaders, optimizer, criterion, device)
        if epoch == 0:
            train_loss, train_acc = val_loss, val_acc

        if config.use_tensorboard:
            writer.add_scalar('train_loss'.format(phase), train_loss, epoch)
            writer.add_scalar('train_acc'.format(phase), train_acc, epoch)
            writer.add_scalar('val_loss'.format(phase), val_loss, epoch)
            writer.add_scalar('val_acc'.format(phase), val_acc, epoch)
            writer.flush()

        print('epoch={} train_loss: {:.3f} train_acc: {:.3f} val_loss: {:.3f} val_acc: {:.3f}'
            .format(epoch, train_loss , train_acc, val_loss , val_acc))
     
        log_epoch = [{'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}]   
        df = pd.DataFrame(log_epoch)
        log_file = '{}/output{}.csv'.format(config.result_path, config.file_name_label)
        if epoch == 0:
            df.to_csv(log_file)
        else:
            df.to_csv(log_file, mode='a', header=False)

        if config.epochs_save != -1 and epoch%config.epochs_save == 0 and epoch != 0:
            torch.save(model.state_dict(), '{}/weight{}_epoch{}.pth'.format(config.result_path, config.file_name_label, epoch))

    #test(model, dataloaders, device)

    print('Finished Training')

    PATH = '{}/weight{}_final.pth'.format(config.result_path, config.file_name_label)
    torch.save(model.state_dict(), PATH)
    if config.save_onnx:
        input = torch.randn(shape)
        input = input.unsqueeze(0)
        print("rand: {}".format(input))
        torch.onnx.export(model, input, '{}/weight_final.onnx'.format(config.result_path), verbose=True)

    if config.use_tensorboard:
        writer.close()


if __name__ == '__main__':

    main()
