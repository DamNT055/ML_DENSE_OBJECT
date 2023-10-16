import os
import torch
import wandb
from datetime import datetime
from pathlib import Path
from model_utils import model_generator
from dataloader import dataloader_generator
from vision_utils.engine import train_one_epoch, evaluate

wandb.init(
    project="ml_dense_object",
    config={
        "learning_rate": 0.0001,
        "architecture": "retinanet_mobilevit",
        "dataset": "SKU110K",
        "epochs": 10
        }
)

PATH = os.path.join(os.fspath(Path(__file__).resolve().parents[0]), 'out_checkpoints')

def main(num_epochs = 1,):
    data_loader, data_loader_test = dataloader_generator()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model,_,_ = model_generator()
    model.to(device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(num_epochs):
        train_one_epoch(model=model, optimizer=optimizer, data_loader=data_loader, device=device, epoch=epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device, loss_only=True)
        torch.save(model.state_dict(), os.path.join(PATH, 'cp-epoch_' + str(epoch)+ '_'+ str(datetime.now()).replace(' ', '_') +'.pt'))   

wandb.finish()

if __name__ == "__main__": 
    main()
    
    