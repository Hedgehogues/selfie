import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class Selfie(nn.Module):
    def __init__(self, resnet, tsize):
        super(Selfie, self).__init__()
        self.tsize = tsize
        self.resnet_cut = nn.Sequential(*list(resnet.children())[:-4])
        self.patch_size = 9
        self.fc_n = 512
        self.fc1_n = self.fc_n * self.patch_size
        self.fc2_n = self.fc_n * int(self.patch_size / 3)
        self.fc3_n = self.fc_n * int(self.patch_size * 2 / 3)
        self.fc_out_1 = self.fc_n

        self.fc1 = nn.Linear(self.fc1_n, self.fc2_n)
        self.fc2 = nn.Linear(self.fc2_n, self.fc3_n)
        self.fc3 = nn.Linear(self.fc3_n, self.fc_out_1)

    def attention0(self, v, n):
        u = v[:, :n].reshape(-1, self.fc1_n)
        u = F.relu(self.fc1(u))
        u = F.relu(self.fc2(u))
        u = F.relu(self.fc3(u))
        return u

    def forward(self, decoder, encoder, target_patch):
        decoder_n = decoder.shape[1]
        encoder_n = encoder.shape[1]

        decoder = decoder.permute(1, 0, 2, 3, 4)
        encoder = encoder.permute(1, 0, 2, 3, 4)
        target_patch = target_patch.permute(1, 0, 2, 3, 4)
        flatten_input = decoder.unbind() + encoder.unbind() + target_patch.unbind()
        x = torch.stack(flatten_input).reshape(-1, 3, 8, 8)
        n = decoder_n + encoder_n + 1
        v = self.resnet_cut(x).view(-1, self.fc_n).reshape(n, -1, self.fc_n)
        v = v.permute(1, 0, 2)
        h0 = v[:, decoder_n + encoder_n:decoder_n + encoder_n + 1].view(-1, self.fc_n)
        h = v[:, decoder_n:decoder_n + encoder_n]

        a = self.attention0(v, decoder_n)
        u = h0 + a
        ts = []
        for i in range(encoder_n):
            t = (u @ h[:, i, :].transpose(0, 1)).diag()
            ts.append(t)
        ts = torch.stack(ts).transpose(0, 1)
        return ts


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


def patches_generator(loader, patch_size=8, decoder_size=9, encoder_size=3):
    for batch in loader:
        decoder = []
        encoder = []
        targets = []
        target_patches = []
        for item in batch[0]:
            decoder_samples = []
            for i in range(decoder_size):
                x = np.random.randint(0, item.shape[1] - patch_size)
                y = np.random.randint(0, item.shape[2] - patch_size)
                decoder_samples.append(item[:, x:x + patch_size, y:y + patch_size])
            encoder_samples = []
            for i in range(encoder_size):
                x = np.random.randint(0, item.shape[1] - patch_size)
                y = np.random.randint(0, item.shape[2] - patch_size)
                encoder_samples.append(item[:, x:x + patch_size, y:y + patch_size])
            target = np.random.randint(encoder_size)
            targets.append(target)
            target_patches.append(torch.stack([encoder_samples[target]]))
            encoder.append(torch.stack(encoder_samples))
            decoder.append(torch.stack(decoder_samples))
        item = torch.stack(decoder), torch.stack(encoder), torch.stack(target_patches), torch.tensor(targets)
        yield item


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
encoder_size = 3
resnet50 = torchvision.models.resnet50(pretrained=False)
resnet50 = resnet50.to(device)
net = Selfie(resnet=resnet50, tsize=encoder_size)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.0
    min_loss = 10000
    max_loss = -10000
    print_step = 10
    torch.autograd.set_detect_anomaly(True)
    for i, (decoder, encoder, target_patch, targets) in enumerate(patches_generator(trainloader, encoder_size=encoder_size)):
        optimizer.zero_grad()
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        target_patch = target_patch.to(device)
        targets = targets.to(device)
        outputs = net(decoder, encoder, target_patch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        min_loss = min(min_loss, loss.item())
        max_loss = max(max_loss, loss.item())
        if i > 0 and i % print_step == 0:
            print('[%d, %5d] loss: %.3f. min_loss: %.3f. max_loss: %.3f' %
                  (epoch + 1, i + 1, train_loss / print_step, min_loss, max_loss))
            train_loss = 0.0
            min_loss = 10000
            max_loss = -10000


def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (decoder, encoder, target_patch, targets) in enumerate(patches_generator(testloader, encoder_size=encoder_size)):
            decoder = decoder.to(device)
            encoder = encoder.to(device)
            target_patch = target_patch.to(device)
            targets = targets.to(device)
            outputs = net(decoder, encoder, target_patch)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(loss.item(), (targets == predicted).type(torch.DoubleTensor).sum() / predicted.shape[0])


for epoch in range(0, 0+200):
    train(epoch)
    test()
