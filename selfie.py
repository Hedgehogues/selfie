import os

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class Attention(nn.Module):
    """
    Computes a weighted average of channels across timesteps (1 parameter pr. channel).

    https://gist.github.com/thomwolf/dec72992ea6817290273d42f6b95c04c
    https://github.com/thomlake/pytorch-attention
    """
    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = nn.Parameter(torch.FloatTensor(attention_size))

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if torch.cuda.is_available():
            idxes = idxes.cuda()
        mask = torch.Variable((idxes < input_lengths.unsqueeze(1)).float())

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        attentions = attentions if self.return_attention else None
        return representations, attentions


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
        self.bn3 = nn.BatchNorm1d(self.fc_out_1)

        self.fc_4_n = self.fc_n * (self.tsize + 1)
        self.fc_5_n = int(self.fc_n / 2 * (self.tsize + 1))
        self.fc_6_n = int(self.fc_n / 4 * (self.tsize + 1))
        self.fc_7_n = int(self.fc_n / 9 * (self.tsize + 1))
        self.fc_out_2 = tsize

        self.fc4 = nn.Linear(self.fc_4_n, self.fc_5_n)
        self.fc5 = nn.Linear(self.fc_5_n, self.fc_6_n)
        self.fc6 = nn.Linear(self.fc_6_n, self.fc_7_n)
        self.fc7 = nn.Linear(self.fc_7_n, self.fc_out_2)
        self.bn_out = nn.BatchNorm1d(self.fc_out_2)

    def fake_attention(self, v, n):
        u = v[:, :n].reshape(-1, self.fc1_n)
        u = F.relu(self.fc1(u))
        u = F.relu(self.fc2(u))
        u = F.relu(self.fc3(u))
        u = self.bn3(u)
        return u

    def forward(self, decoder, encoder, target_patch):
        decoder_n = decoder.shape[1]
        encoder_n = encoder.shape[1]
        batch_size = encoder.shape[0]

        decoder = decoder.permute(1, 0, 2, 3, 4)
        encoder = encoder.permute(1, 0, 2, 3, 4)
        target_patch = target_patch.permute(1, 0, 2, 3, 4)
        flatten_input = decoder.unbind() + encoder.unbind() + target_patch.unbind()
        x = torch.stack(flatten_input).reshape(-1, 3, 8, 8)
        n = decoder_n + encoder_n + 1
        v = self.resnet_cut(x).view(-1, self.fc_n).reshape(-1, self.fc_n).reshape(-1, n, self.fc_n)

        u = self.fake_attention(v, decoder_n)
        h0 = v[:, decoder_n+encoder_n:decoder_n+encoder_n+1].view(-1, self.fc_n)
        h = v[:, decoder_n:decoder_n + encoder_n].view(batch_size, -1)
        res = torch.cat([u + h0, h], dim=1)
        res = F.relu(self.fc4(res))
        res = F.relu(self.fc5(res))
        res = F.relu(self.fc6(res))
        res = self.fc7(res)
        res = self.bn_out(res)

        return res


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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

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
resnet50 = torchvision.models.resnet50(pretrained=True)
resnet50 = resnet50.to(device)
net = Selfie(resnet=resnet50, tsize=encoder_size)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)


# Training
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


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
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

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


for epoch in range(0, 0+200):
    train(epoch)
    test(epoch)
