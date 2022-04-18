import torch
from math import prod
from models.backbone import BackBoneNN
from torch.distributions import Categorical, Normal

class PPO_model(torch.nn.Module):
    def __init__(self, input_width: int, input_height: int, action_space, is_discrete:bool):
        super(PPO_model, self).__init__()
        self.is_discrete = is_discrete
        self.width = input_width
        self.height = input_height
        self.action_space = action_space
        self.preprocessor = BackBoneNN(input_width=input_width, input_height=input_height)

        if is_discrete:
            # TODO!!!: Haven't dealt with discrete action spaces yet have see what's appropriate
            self.logits_net = torch.nn.Sequential(torch.nn.Linear(in_features=self.preprocessor.out_len, out_features=len(action_space)))
        else:
            self.action_len = prod(self.action_space._shape)
            self.mu_net = torch.nn.Sequential(torch.nn.Linear(in_features=self.preprocessor.out_len, out_features=self.action_len))
            self.sd_logits_net = torch.nn.Sequential(torch.nn.Linear(in_features=self.preprocessor.out_len, out_features=self.action_len))

        self.value = torch.nn.Sequential(torch.nn.Linear(in_features=self.preprocessor.out_len, out_features=1))


    def forward(self, obs: torch.Tensor):
        """
            Forwards RGB obs through neural network designed for PPO

            Returns:
                pi: Policy distribution
                v: Predicted value of obs/state
        """
        out_pre = self.preprocessor.forward(obs)
        
        if self.is_discrete:
            out_logits_net = self.logits_net(out_pre)
            pi = Categorical(logits=out_logits_net)
        else:
            mu = self.mu_net(out_pre)
            std = torch.exp(self.sd_logits_net(out_pre))
            pi = Normal(loc=mu, scale=std)
        
        v = self.value.forward(out_pre)

        return pi, v
