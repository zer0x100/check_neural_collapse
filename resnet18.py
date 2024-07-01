""" Train ResNet18 for MNIST during TPT(Terminal Phase Training) for investigating Neural Collapse.
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.auto import tqdm

#GPUが利用可能か
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#保存先作成
os.makedirs('./trained_2', exist_ok=True)
OUTNAME = './trained_2/OUT'

#parameters
batch_size = 128
num_epochs = 200
#SGD params
lerning_rate = 0.1
weight_decay = 0.0001
momentum = 0.9

#データの前処理と読み込み
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#ResNet18モデルの読み込みと最終層置換
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

#モデルをGPUに移動
model = model.to(device)

#損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lerning_rate, weight_decay=weight_decay, momentum=momentum)

#各エポック毎、各クラス毎の入力の平均値を記録する
train_class_ftrs_mean = []
test_class_ftrs_mean = []
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(1, num_epochs+1):
    #Switch to training mode
    model.train()
    #Pre-minibatch training
    loss_sum = 0
    correct_sum = 0
    input_count = 0
    class_ftrs_sum = torch.zeros(10, num_ftrs).to(device)
    class_counts = torch.zeros(10).to(device)
    bar = tqdm(train_loader, desc='Train Epoch: %d' % epoch, unit='batches')
    for inputs, labels in bar:
        input_count += len(inputs)
        inputs, labels = inputs.to(device), labels.to(device)
        #updata step
        first_conv_out = model.maxpool(model.relu(model.bn1(model.conv1(inputs))))
        features = torch.flatten(model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(first_conv_out))))), 1)
        outputs = model.fc(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #show losses and accuracies
        loss_sum += loss.item() * len(inputs)
        pred = outputs.argmax(dim=1)
        correct_sum += pred.eq(labels).sum().item()
        bar.set_postfix(loss_avg=loss_sum / input_count, accuracy=correct_sum / input_count)
        #各クラスの入力の合計とカウントを計算
        for i in range(inputs.size(0)):
            class_index = labels[i]
            class_ftrs_sum[class_index] += features[i]
            class_counts[class_index] += 1
    train_losses.append(loss_sum/input_count)
    train_accuracies.append(100*correct_sum/input_count)
    class_ftrs_mean = class_ftrs_sum / class_counts.unsqueeze(1)
    train_class_ftrs_mean.append(class_ftrs_mean.cpu().detach().numpy())

    #Switch to inference mode
    model.eval()
    loss_sum = 0
    correct_sum = 0
    input_count = 0
    class_ftrs_sum = torch.zeros(10, num_ftrs).to(device)
    class_counts = torch.zeros(10).to(device)
    with torch.no_grad(): #We do not nee auto grad for inference
        bar = tqdm(test_loader, desc='Test Epoch: %d' % epoch, unit='batches')
        for inputs, labels in bar:
            input_count += len(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            first_conv_out = model.maxpool(model.relu(model.bn1(model.conv1(inputs))))
            features = torch.flatten(model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(first_conv_out))))), 1)
            outputs = model.fc(features)
            loss_sum += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1)
            correct_sum += pred.eq(labels).sum().item()
            bar.set_postfix(loss_avg=loss_sum / input_count, accuracy=correct_sum/input_count)
            for i in range(inputs.size(0)):
                class_index = labels[i]
                class_ftrs_sum[class_index] += features[i]
                class_counts[class_index] += 1
    test_loss, test_accuracy = loss_sum/input_count, 100*correct_sum/input_count

    #save the model if it it better than before
    if len(test_losses) > 1 and test_loss < min(test_losses):
        torch.save(model.state_dict(), f'{OUTNAME}_best_loss.tar')
    if len(test_accuracies) > 1 and test_accuracy < min(test_accuracies):
        torch.save(model.state_dict(), f'{OUTNAME}_best_accuracy.tar')
    
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    class_ftrs_mean = class_ftrs_sum / class_counts.unsqueeze(1)
    test_class_ftrs_mean.append(class_ftrs_mean.cpu().detach().numpy())
    
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

torch.save(model.state_dict(), f'{OUTNAME}.tar')

results = {
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'train_class_ftrs_mean': train_class_ftrs_mean,
    'test_class_ftrs_mean': test_class_ftrs_mean,
}
np.savez(f'{OUTNAME}_results.npz', **results)
