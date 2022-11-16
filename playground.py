import torch
import ot
lambd = 1e-3
Gs = ot.sinkhorn(x, y, M, lambd, verbose=True)

x = torch.tensor([[1.0, 0.0, 0.0, 0.0,0.0]])
y = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])

loss = SamplesLoss(loss="sinkhorn")
print(loss(x, y))