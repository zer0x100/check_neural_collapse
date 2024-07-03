import torch
from torch import nn, optim
import torch.utils
import torch.utils.data
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
import os
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

#Define a custom dataset class for office-31 dataset
class Office31Dataset(torch.utils.data.Dataset):
    def __init__(self, root, domain, transform=None):
        self.root = root
        self.domain = domain
        self.transform = transform
        self.samples = self.make_dataset()
    
    def make_dataset(self):
        domain_path = os.path.join(self.root, self.domain)
        self.classes = [d.name for d in os.scandir(domain_path) if d.is_dir()]
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(domain_path, class_name)
            for root, _, fnames in sorted(os.walk(class_path)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[class_name])
                    samples.append(item)
        return samples
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __len__(self):
        return len(self.samples)

def main():
    #parameters
    batch_size = 32
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.0001
    num_epochs = 200
    os.makedirs("./trained_office", exist_ok=True)
    OUTNAME = "./trained_office/OUT" #local path where the data will be saved

    #Define the transforms for the training and validation datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    #Load the Office-31 dataset
    data_dir = 'Office_31' # path to Office-31 datasets in my local PC
    train_domain = 'amazon'
    val_domain = 'webcam'

    train_dataset = Office31Dataset(root=data_dir, domain=train_domain, transform=data_transforms['train'])
    val_dataset = Office31Dataset(root=data_dir, domain=val_domain, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    #Load a pretrained ResNet18 model and modify the final layer
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    nclasses = len(train_dataset.classes)
    model.fc = nn.Linear(num_ftrs, nclasses)

    #Move the model to the appopriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    #Train the model
    dataloaders = {'train': train_loader, 'val': val_loader}
    class_ftrs_mean = {'train': [], 'val': []}
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        #Training phase and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode
            
            bar = tqdm(dataloaders[phase], desc=f'{phase} Epoch: {epoch}', unit='batches')

            loss_sum = 0
            correct_sum = 0
            input_count = 0
            class_ftrs_sum = torch.zeros(nclasses, num_ftrs).to(device)
            class_counts = torch.zeros(nclasses).to(device)

            for inputs, labels in bar:
                inputs, labels = inputs.to(device), labels.to(device)
                input_count += len(inputs)

                #Forward
                with torch.set_grad_enabled(phase == 'train'):
                    first_conv_out = model.maxpool(model.relu(model.bn1(model.conv1(inputs))))
                    features = torch.flatten(model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(first_conv_out))))), 1)
                    outputs = model.fc(features)
                    loss = criterion(outputs, labels)
                    #update
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    #show losses and accuracies
                    loss_sum += loss.item() * len(inputs)
                    pred = outputs.argmax(dim=1)
                    correct_sum += pred.eq(labels).sum().item()
                    bar.set_postfix(loss_avg=loss_sum / input_count, accuracy=correct_sum / input_count)
                    #add last-layer's features to features-sum for each class
                    for i in range(inputs.size(0)):
                        class_index = labels[i]
                        class_ftrs_sum[class_index] += features[i]
                        class_counts[class_index] += 1
            loss, accuracy = loss_sum / input_count, 100*correct_sum / input_count

            #save model if it is better than before
            if phase == 'val':
                if len(losses['val']) > 1 and loss < min(losses['val']):
                    torch.save(model.state_dict(), f'{OUTNAME}_best_loss.tar')
                if len(accuracies['val']) > 1 and accuracy > min(accuracies['val']):
                    torch.save(model.state_dict(), f'{OUTNAME}_best_accuracy.tar')

            #save loss, accuracy, and each class features mean
            losses[phase].append(loss)
            accuracies[phase].append(accuracy)
            class_ftrs_mean[phase].append((class_ftrs_sum / class_counts.unsqueeze(1)).cpu().detach().numpy())

        #anneal learning rate
        if epoch == num_epochs // 3 or epoch == 2 * num_epochs // 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

    torch.save(model.state_dict(), f'{OUTNAME}.tar')

    results = {
        'train_losses': losses['train'],
        'train_accuracies': accuracies['train'],
        'test_losses': losses['val'],
        'test_accuracies': accuracies['val'],
        'train_class_ftrs_mean': class_ftrs_mean['train'],
        'test_class_ftrs_mean': class_ftrs_mean['val'],
    }
    np.savez(f'{OUTNAME}_results.npz', **results)

if __name__ == '__main__':
    main()