param(
    [string]$EnvName = "mamba_env",
    [string]$TritonSpec = "triton-windows==3.6.0.post26"
)

$ErrorActionPreference = "Stop"

Write-Host "Preparing Triton for Mamba3 in micromamba env '$EnvName'"
Write-Host "Installing: $TritonSpec"

micromamba run -n $EnvName python -m pip uninstall -y triton triton-windows
micromamba run -n $EnvName python -m pip install -U pip setuptools wheel
micromamba run -n $EnvName python -m pip install -U "$TritonSpec"

$TritonCache = Join-Path $env:USERPROFILE ".triton\cache"
$TorchInductorCache = Join-Path $env:TEMP "torchinductor_$env:USERNAME"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $TritonCache
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $TorchInductorCache

$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:TRITON_CACHE_DIR = Join-Path $RepoRoot ".triton_cache\setup_test"
New-Item -ItemType Directory -Force -Path $env:TRITON_CACHE_DIR | Out-Null

$TestScript = @'
import inspect
import torch
import triton

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__, triton.__file__)
print("Config", inspect.signature(triton.Config))
print("has set_allocator", hasattr(triton, "set_allocator"))

from mamba_ssm import Mamba3

model = Mamba3(d_model=128, d_state=64, expand=2, headdim=64, ngroups=1, chunk_size=64).cuda()
x = torch.randn(2, 16, 128, device="cuda")
y = model(x)
loss = y.float().square().mean()
loss.backward()
print("Mamba3 forward/backward ok", tuple(y.shape), float(loss.detach().cpu()))
'@

$Tmp = Join-Path $env:TEMP "test_mamba3_triton_windows.py"
Set-Content -Encoding ASCII -Path $Tmp -Value $TestScript
micromamba run -n $EnvName python $Tmp

Write-Host "Mamba3 Triton setup finished."
