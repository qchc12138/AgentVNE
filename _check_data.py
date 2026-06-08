import torch
d = torch.load('e:/E桌面/AgentVNE/pretrain_data/pretrain_dataset.pt', map_location='cpu', weights_only=False)
print('Dataset keys:', list(d.keys()))
print('Num samples:', len(d['samples']))
s = d['samples'][0]
print('Sample keys:', list(s.keys()))
print('label shape:', s['label'].shape)
print('wf nodes:', s['workflow_graph'].x.shape)
print('sn nodes:', s['substrate_graph'].x.shape)
