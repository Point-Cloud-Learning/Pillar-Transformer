import math
import numpy as np
from torch import optim
from torch.optim import lr_scheduler
from MyDataSet import MyDataSet, Compose, ToTensor
from PillarTransformer import pit_base_patch_point1
from torch.utils.data import DataLoader
from Utils import train_one_epoch, evaluate
import torch


def lr_lambda(x):
    if x < 20:
        return math.pow(math.pow(10, 1 / 19), x)  # Warmup
    else:
        return 10 * math.pow(math.pow(0.1, 1 / 180), x - 19)  # Decay


def lr_lambda_Cosine(x):
    return (0.001 + (0.01 - 0.001) * (1 + math.cos(math.pi * (x + 108) / 200)) / 2) * 100


if __name__ == "__main__":
    batch_size = 32
    lr = 0.001
    epochs = 200
    num_classes = 15

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_file_path = "./ScanObjectNN_training_bg.h5"
    test_file_path = "./ScanObjectNN_testing_bg.h5"

    train_dataset = MyDataSet(train_file_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, drop_last=True)

    test_dataset = MyDataSet(test_file_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)

    model = pit_base_patch_point1(num_classes=num_classes).to(device)
    # model.load_state_dict(torch.load("./test2/model-107.pth"))

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5E-5)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(0, epochs):
        # train
        train_loss, train_acc = np.around(train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch), 5)

        # validate
        val_loss, val_acc = np.around(evaluate(model=model, data_loader=test_loader, device=device, epoch=epoch), 5)

        lr_c = optimizer.param_groups[0]["lr"]
        tag = f"epoch {epoch}  train_loss: {train_loss}, train_acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}, lr: {lr_c} "
        print(tag)

        scheduler.step()

        torch.save(model.state_dict(), "./test/model-{}.pth".format(epoch))
        with open("./test/record_per_epoch.txt", 'a') as f:
            f.write(tag + '\n')
