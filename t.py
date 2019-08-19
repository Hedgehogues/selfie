import torch
from torch import nn


class Nn(nn.Module):
    def __init__(self):
        super(Nn, self).__init__()
        self.ln = nn.Linear(5, 5)

    def forward(self, x):
        v = self.ln(x)

        u = v.clone()
        h = v.clone()

        u /= u.norm()
        h = h.detach()
        h /= h.norm()

        res = torch.stack([torch.stack([u @ h, u @ h])])

        return res


def patches_generator():
    while True:
        decoder = torch.rand((5, ))
        target = torch.randint(2, (1,))
        yield decoder, target


net = Nn()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

net.train()
torch.autograd.set_detect_anomaly(True)
for decoder, targets in patches_generator():
    optimizer.zero_grad()
    outputs = net(decoder)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

