import torch


class _SemanticCorrectionGate(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(in_channels), int(hidden_channels)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(int(hidden_channels), 1),
        )

    def forward(self, features):
        original_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        return self.net(flat).reshape(*original_shape)


class _SemanticCorrectionDeferGate(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.candidate_scorer = torch.nn.Sequential(
            torch.nn.Linear(int(in_channels), int(hidden_channels)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(int(hidden_channels), 1),
        )
        self.base_scorer = torch.nn.Sequential(
            torch.nn.Linear(int(in_channels) * 2, int(hidden_channels)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(int(hidden_channels), 1),
        )

    def forward(self, features):
        candidate_logits = self.candidate_scorer(features).squeeze(-1)
        pooled = torch.cat([features.mean(dim=1), features.max(dim=1).values], dim=1)
        base_logit = self.base_scorer(pooled)
        return torch.cat([base_logit, candidate_logits], dim=1)


class _SemanticBreakRiskHead(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(in_channels), int(hidden_channels)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(int(hidden_channels), 1),
        )

    def forward(self, features):
        original_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        return self.net(flat).reshape(*original_shape)
