
import multiprocessing
import torch
from torch import optim
from faster_rcnn import FasterRCNN


if __name__ == '__main__':
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 11,
        'pretrained': True,
        'num_epochs': 10,
        'batch_size': 2,
        'learning_rate': 4e-5,
        'weight_decay': 5e-5,
        'optimizer': optim.AdamW,
        'log_dir': 'log/visualization',
    }
    multiprocessing.freeze_support()
    model = FasterRCNN(config)
    #model.load_model("model.pth")
    model.train()
    #model.eval()
    #model.visual()
    #model.test()
