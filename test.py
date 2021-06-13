import torch

action_dim = 3
new_action_std = 0.5
var = torch.full((action_dim,), new_action_std * new_action_std)
print(var)
cov_mat = torch.diag(var)
print(cov_mat)
cov_mat = cov_mat.unsqueeze(dim=0)
print(cov_mat)