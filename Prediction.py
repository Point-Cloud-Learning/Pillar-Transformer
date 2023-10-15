import h5py
import numpy as np

from MyDataSet import MyDataSet, Compose, ToTensor
from PillarTransformer import pit_base_patch_point1
import torch


def main(index, data, label, model_weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor()])
    data, label = transform(data, label)
    label = label.numpy()

    data = torch.unsqueeze(data, dim=0)

    model = pit_base_patch_point1(num_classes=40).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(data.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "第{}项: 真实类 {}, 预测类 {}, 预测类概率 {:.3}, 预测 {}".format(index, label, predict_cla, predict[predict_cla].numpy(), predict_cla == label)
    print(print_res)


if __name__ == '__main__':
    test_file_path = "D:/PointCloudEncoding/Project_one/Modelnet40_test.h5"
    file = h5py.File(test_file_path, 'r')
    data = np.array(file["data"])
    label = np.array(file["label"])

    main(0, data[0], label[0], "./x_y_division_weights/model-8.pth")

