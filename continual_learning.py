import torch
import torch.nn as nn

class KnowledgeDistillationLoss(nn.Module):
    """Prevents forgetting old classes while learning new ones"""
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, old_outputs, new_outputs):
        old_soft = torch.softmax(old_outputs / self.temperature, dim=1)
        new_soft = torch.log_softmax(new_outputs / self.temperature, dim=1)
        return self.kl_div(new_soft, old_soft) * (self.temperature ** 2)

class EWC:
    """Elastic Weight Consolidation — protects important weights"""
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            output = self.model(batch.to(self.device))
            loss = output.sum()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss

class ReplayBuffer:
    """Stores exemplars from old classes to prevent forgetting"""
    def __init__(self, max_size=200):
        self.buffer = []
        self.max_size = max_size

    def add(self, sample):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(sample)

    def get_samples(self, n=32):
        import random
        return random.sample(self.buffer, min(n, len(self.buffer)))
