import os
import logging
import torch
from datetime import datetime
from model_utils import model_generator
from dataloader import dataloader_generator
from vision_utils.engine import train_one_epoch, evaluate
from pathlib import Path


import torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

logger = logging.getLogger(__name__)
PATH = os.path.join(os.fspath(Path(__file__).resolve().parents[0]), 'out_checkpoints')

def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def clean_up():
    dist.destroy_process_group()

def main(num_epochs = 1):
    ddp_setup()
    data_loader, data_loader_test = dataloader_generator()
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = int(os.environ["LOCAL_RANK"])
    model,_,_ = model_generator()
    model.to(device=device)
    model = DDP(model, device_ids=[device])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(num_epochs):
        train_one_epoch(model=model, optimizer=optimizer, data_loader=data_loader, device=device, epoch=epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), os.path.join(PATH, 'cp-epoch_' + str(epoch)+ '_'+ str(datetime.now()).replace(' ', '_') +'.pt'))   
    destroy_process_group()

if __name__ == "__main__": 
    try: 
        world_size = torch.cuda.device_count()
        print("World size: ", world_size)
        main()
    except Exception as e:
        logger.error(e)
        clean_up()
    
    