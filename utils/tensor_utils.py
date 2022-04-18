import torch

def process_obs(obs: torch.tensor, model):
    if not isinstance(obs, torch.Tensor):
        try:
            obs = torch.tensor(obs)
        except ValueError as e:
            obs = torch.tensor(obs.copy())
    
    if obs.dtype == torch.uint8:
        obs = obs/255
    elif obs.dtype != torch.float32:
        raise ValueError(f'obs is of tensor dtype={obs.dtype}, but should be: {torch.float32}')

    if len(obs.shape) != 4:
        print('obs should be of dimension Batch x Num Channel x Height x Width. Creating 4th dimension...')
        obs = torch.unsqueeze(obs, 0)
    
    if obs.shape[1] != 3:
        if obs.shape[3] == 3:
            print('obs should be of Batch x 3 x Height x Width. Transposing...')
            obs = torch.transpose(obs, 1, 3)
        else:
            raise ValueError('obs first dimension should be size 3 for RGB channels')
    
    if obs.shape[2] != model.height:
        raise ValueError(f'obs height should be {model.height}')

    if obs.shape[3] != model.width:
        raise ValueError(f'obs width should be {model.width}')
    
    return obs