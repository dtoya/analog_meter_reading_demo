import torch
import models

model = models.alexnet.get_model(num_classes=1)
model_path = './result/weight_final.pth'
model.load_state_dict(torch.load(model_path))
input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input, './result/weight_final.onnx', verbose=True)
