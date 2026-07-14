import torch
from pathlib import Path
data = torch.load(Path.home() / '.cache' / 'nops' / 'navier_stokes_v1e-3_N1200_T20.pt').float()
print(f'Data shape: {data.shape}')
print()

# Check one-step dynamics at different timesteps
for t in [0, 5, 10, 15, 18]:
    diffs = []
    for i in range(100):
        diff = (data[i, :, :, t+1] - data[i, :, :, t]).float().pow(2).mean().sqrt().item()
        diffs.append(diff)
    mean_diff = sum(diffs)/len(diffs)
    f0 = data[i, :, :, t].min().item()
    f1 = data[i, :, :, t].max().item()
    print(f't={t}->{t+1}: |delta|=mean={mean_diff:.4f}  range=[{f0:.2f}, {f1:.2f}]')
    
# L2 norm of prediction target
for t in [0, 5, 10, 15, 18]:
    norms = [data[i, :, :, t+1].float().pow(2).mean().sqrt().item() for i in range(100)]
    mn = sum(norms)/len(norms)
    sd = ((sum((n-mn)**2 for n in norms)/len(norms))**0.5)
    print(f't={t+1}: ||vort||_L2 = {mn:.4f} +/- {sd:.4f}')
