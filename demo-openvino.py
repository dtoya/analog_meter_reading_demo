from openvino.inference_engine import IECore
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

    ie = IECore()

    log.info('Read Network: {}'.format(args.model))
    net = ie.read_network(model=args.model)

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    log.info('Load Network: Device = {}'.format(args.device))
    exec_net = ie.load_network(network=net, device_name=args.device)

    input_shape = net.input_info[input_blob].input_data.shape
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
        res = exec_net.infer(inputs={input_blob: image})
        res = res[out_blob][0][0]
        value = (res+0.5)*2000
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

