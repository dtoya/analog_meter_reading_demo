

### Download dataset
```
$ ./download-dataset.sh
```

```
$ python3 -m venv venv
$ . venv/bin/activate
(venv) $ pip install -U pip
(venv) $ pip install -r requirements.txt
```

```
(venv) $ python train.py --config_file configs/config-alexnet.json
```

```
(venv) $ python demo-pytorch.py -w result/weight_final.pth -i ./dataset/analog-meter/istockphoto-105940980-640_adpp_is.mp4 --input_size 224 224 --model_name alexnet
```

```
(venv) $ python convert-to-onnx.py
(venv) $ sh convert-to-ir.sh
(venv) $ python demo-openvino.py -m ./result/FP16/analog-meter.xml -i ./dataset/analog-meter/istockphoto-105940980-640_adpp_is.mp4 -d CPU
```

