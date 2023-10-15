import sys
import torch
from tqdm import tqdm
import h5py
import numpy as np
from MyDataSet import Compose, ToTensor


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    transform = Compose([ToTensor()])
    optimizer.zero_grad()

    sample_num = 0
    # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        images = images.data.numpy()
        images, labels = transform(images, labels)
        images, labels = images.to(device), labels.to(device)

        sample_num += images.shape[0]

        predict = model(images)
        predict_classes = torch.max(predict, dim=1)[1]

        accu_num += torch.eq(predict_classes, labels).sum()

        loss = loss_function(predict, labels)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    transform = Compose([ToTensor()])
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    num_per_class = torch.as_tensor([0 for _ in range(model.num_classes)], dtype=torch.float32).to(device)  # 累计每一类的样本数
    right_num_per_class = torch.as_tensor([0 for _ in range(model.num_classes)], dtype=torch.float32).to(device)  # 累计每一类的正确分类样本数

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        images = images.data.numpy()
        # images = normalize_data(images)
        images, labels = transform(images, labels)
        images, labels = images.to(device), labels.to(device)

        sample_num += images.shape[0]

        predict = model(images)
        predict_classes = torch.max(predict, dim=1)[1]

        accu_num += torch.eq(predict_classes, labels).sum()

        num_per_class += torch.bincount(labels, minlength=model.num_classes)
        right_num_per_class += torch.bincount(labels, weights=labels == predict_classes, minlength=model.num_classes)

        loss = loss_function(predict, labels)
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    right_rate_per_class = ' '.join([str(x) for x in np.around(np.array((right_num_per_class / num_per_class).cpu()), 3)])
    num_per_class = ' '.join([str(x) for x in np.around(np.array(num_per_class.cpu()), 3)])
    right_num_per_class = ' '.join([str(x) for x in np.around(np.array(right_num_per_class.cpu()), 3)])
    with open("./test/valid_each_class_per_epoch.txt", 'a') as f:
        f.write(
            f"epoch {epoch}\n  每个类的参与总数：{num_per_class}\n  每个类正确分类数：{right_num_per_class}\n  每个类的正确比例：{right_rate_per_class}\n")

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
