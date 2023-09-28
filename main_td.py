import torch 
from options.opts import get_training_arguments
from cvnets import get_model

opts = get_training_arguments()
setattr(opts, "model.detection.n_classes", 81)
model = get_model(opts)
encoder = model.encoder
print(encoder)
exit()
for e in dir(model):
    print(e)