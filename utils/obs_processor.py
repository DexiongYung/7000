import torch

def process_obs(obs: torch.tensor):
    if not isinstance(obs, torch.Tensor):
        try:
            obs = torch.tensor(obs)
        except ValueError as e:
            obs = torch.tensor(obs.copy())

    # Convert int tensor to float
    if obs.dtype == torch.uint8:
        obs = obs/255
    
    if obs.dtype != torch.float32:
        raise ValueError(f'obs is of tensor dtype={obs.dtype}, but should be: {torch.float32}')
    
    err_sz = 'obs should be of dimension Batch x Num Channel x Height x Width.'
    if len(obs.shape) == 3:
        # print(f'{err_sz} Creating 4th dimension...')
        obs = torch.unsqueeze(obs, 0)
    elif len(obs.shape) != 4:
        raise ValueError(f'{err_sz} Got obs of shape length = {len(obs.shape)}.')
    
    if obs.shape[1] != 3:
        if obs.shape[3] == 3:
            # print('obs should be of Batch x 3 x Height x Width. Transposing...')
            obs = torch.transpose(obs, 1, 3)
        else:
            raise ValueError('obs first dimension should be size 3 for RGB channels')
     
    return obs