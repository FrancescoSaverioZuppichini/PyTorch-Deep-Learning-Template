import torch
import time
from torch.nn.functional import softmax
from PIL import Image
from Project import Project
from data.transformation import val_transform
from models import MyCNN, resnet18
from utils import device, show_dl
import onnxruntime as ort
import numpy as np
project = Project()

# dummy_input = torch.randn(1, 3, 224, 224).to(device)
# # load and export our best model
# model = resnet18(2).to(device)
# model.load_state_dict(
#     torch.load(project.checkpoint_dir /
#                'resnet18-finetune-1582189784.2430544-model.pt'))
# model.eval()
# torch.onnx.export(model,
#                   dummy_input,
#                   project.checkpoint_dir /
#                   "resnet18-finetune-1582189784.2430544-model.onnx",
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                 'output' : {0 : 'batch_size'}},
#                   verbose=True)



classes = ['Darth Vader', 'Luke Skywalker']

x = Image.open(project.data_dir / 'val/darth-vader/78.yhL6Pa.jpg')
x = val_transform(x)
x = x.unsqueeze(0).numpy()
ort_session = ort.InferenceSession(str(project.checkpoint_dir /
                  "resnet18-finetune-1582189784.2430544-model.onnx"))

start = time.time()
preds = ort_session.run(None, {'input': x})
end = time.time()
probs = softmax(torch.Tensor(preds), dim=-1)
print(f'Is {classes[torch.argmax(probs)]}, inference time = {end - start:4f}')