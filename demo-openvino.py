from openvino.runtime import Core
import logging 
import sys
import cv2
import numpy as np
from argparse import ArgumentParser

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=str)
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. Path to an image, folder with images, video file or a numeric camera ID.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    return parser

def main():
    args = build_argparser().parse_args()

    core = Core()

    log.info('Read and Load Network: Model = {} Device = {}'.format(args.model, args.device))
    model = core.compile_model(args.model, device_name=args.device)
    output_layer = model.output(0)
    input_shape = model.input(0).shape
    log.info('Input shape: {}'.format(input_shape))

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        log.error('OpenCV: Failed to open capture: ' + str(input_stream))
        sys.exit(1)

    log.info('Press Q/q to stop demo.')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.resize(frame, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1)) 
        image = image.reshape(input_shape) 
        res = model([image])[output_layer]
        value = (res[0][0]+0.5)*2000
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

