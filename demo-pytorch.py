import torch
import torchvision.transforms as transforms
import logging
import sys
import cv2
from PIL import Image
import numpy as np
from argparse import ArgumentParser
import models
import time

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

# For repeatability of result
torch.manual_seed(1234)
#np.random.seed(1234)
#random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-w', '--load_weights', help='Required. Path to an weight file of trained model.',
                      required=True, type=str)
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. Path to an image, folder with images, video file or a numeric camera ID.')
    args.add_argument('-d', '--device', default='cuda:0', type=str,
                      help='Optional. Specify the target device to infer on. Default value is GPU.')
    args.add_argument('--input_size', default=[224,224], nargs='*', help='Optional. Input size = [h, w]')
    args.add_argument('--model_name', default='alexnet', help='Optional. Model name')
    return parser

def get_input_image_size(img, size): # img: cv2/ndarray size=(h, w) or scalar
    if isinstance(size, int) or len(size) == 1:
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return oh, ow
        else:
            oh = size
            ow = int(size * w / h)
            return oh, ow
    else:
        return int(size[0]), int(size[1])
        
def main():
    args = build_argparser().parse_args()

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        if args.device == 'cuda:0':
            print('Warning: CUDA deivce is not available. Use CPU.')
        device = torch.device('cpu')
    log.info('Device ={}'.format(device))

    resize = list(map(int, args.input_size))
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    
    if args.model_name == 'alexnet':
        model_name = models.alexnet
    model = model_name.get_model(num_classes=1)
    model.load_state_dict(torch.load(args.load_weights))
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        log.error('OpenCV: Failed to open capture: ' + str(input_stream))
        sys.exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform(image)

        image = image.unsqueeze(0)
        image = image.to(device)
        outputs = model(image)

        value = outputs[0][0]*2000+1000
        cv2.putText(frame, '{:4.0f}'.format(value), (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 5, cv2.LINE_AA)
        cv2.imshow('Detection Results', frame)
        key = cv2.waitKey(1)

        ESC_KEY = 27
        if key in {ord('q'), ord('Q'), ESC_KEY}:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)

