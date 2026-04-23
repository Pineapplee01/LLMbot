param(
    [string]$HostName = "172.31.106.108",
    [int]$Port = 10011,
    [string]$User = "root",
    [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"

Get-Command ssh -ErrorAction Stop | Out-Null

$remoteScript = @'
set -euo pipefail

source /root/mambaforge/etc/profile.d/conda.sh
conda activate lmbot

cd /root/workspace/LMbot/LLMbot/baseline/core

echo "=== identity ==="
whoami
hostname
pwd

echo "=== gpu ==="
nvidia-smi

echo "=== python ==="
which python
python --version

echo "=== package probe ==="
python -W ignore -c "import importlib.util; import torch, torch_geometric, transformers, sklearn; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('device_count=' + str(torch.cuda.device_count())); print('torch_geometric=' + torch_geometric.__version__); print('transformers=' + transformers.__version__); print('sklearn=' + sklearn.__version__); wandb_spec = importlib.util.find_spec('wandb'); print('wandb=' + ((__import__('wandb').__version__) if wandb_spec is not None else 'OPTIONAL_MISSING'))"

echo "=== dataset ==="
python -c "from pathlib import Path; dataset = 'TwiBot-20'; aliases = [dataset]; aliases += ['TwiBot-20', 'Twibot-20'] if dataset.lower() == 'twibot-20' else []; candidate_roots = [Path('./datasets'), Path('../datasets'), Path('../../datasets'), Path('../../../datasets')]; candidates = [root / alias for root in candidate_roots for alias in aliases]; path = next((candidate for candidate in candidates if candidate.exists()), None); path = path or next((child for root in candidate_roots if root.exists() for child in [{entry.name.lower(): entry for entry in root.iterdir() if entry.is_dir()}.get(dataset.lower())] if child is not None), candidates[0]); required = ['train_idx.pt', 'valid_idx.pt', 'test_idx.pt', 'norm_user_text.json', 'labels.pt', 'edge_index.pt', 'edge_type.pt']; missing = [name for name in required if not (path / name).exists()]; print('resolved_dataset_path=' + str(path)); print('resolved_path_exists=' + str(path.exists())); print('required_files_ok=' + str(not missing)); print('missing_files=' + ('NONE' if not missing else ', '.join(missing)))"
'@

if ($PrintOnly) {
    Write-Output ("ssh -o BatchMode=yes -o ConnectTimeout=10 {0}@{1} -p {2} 'bash -s'" -f $User, $HostName, $Port)
    Write-Output ""
    Write-Output $remoteScript
    exit 0
}

$remoteScript | ssh -o BatchMode=yes -o ConnectTimeout=10 ("{0}@{1}" -f $User, $HostName) -p $Port "bash -s"

if ($LASTEXITCODE -ne 0) {
    throw ("Remote smoke test failed with exit code {0}." -f $LASTEXITCODE)
}
