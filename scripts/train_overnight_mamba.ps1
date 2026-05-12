param(
    [string]$EnvName = "mamba_env",
    [string]$EnvId = "MiniGrid-MemoryS17Random-v0",
    [string]$RunName = "overnight_mamba3_s17random_turbo_masked_seed42",
    [int]$Seed = 42,
    [int]$TotalSteps = 100000000,
    [int]$NumEnvs = 64,
    [int]$NumSteps = 256,
    [int]$ContextLen = 128,
    [int]$ChunkLen = 64,
    [int]$BatchChunks = 64,
    [int]$NEpochs = 4,
    [int]$DModel = 128,
    [string]$SpatialEncoder = "hybrid",
    [int]$MambaLayers = 2,
    [int]$DState = 64,
    [string]$MambaVariant = "mamba3",
    [string]$CudaVisibleDevices = "0",
    [int]$SaveInterval = 50000,
    [int]$EvalInterval = 500000,
    [int]$EvalEpisodes = 20,
    [int]$LogInterval = 1,
    [string]$ValidActions = "0,1,2",
    [double]$EntCoef = 0.01,
    [double]$EntCoefFinal = 0.001,
    [string]$ResumeFrom = "",
    [string]$FallbackVariant = "mamba2",
    [switch]$NoFallback
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$TimeStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $RepoRoot "logs\$RunName`_$TimeStamp.log"

$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
$env:TRITON_CACHE_DIR = Join-Path $RepoRoot ".triton_cache\$RunName"
New-Item -ItemType Directory -Force -Path $env:TRITON_CACHE_DIR | Out-Null
Write-Host "Triton cache: $env:TRITON_CACHE_DIR"

$RequestedVariant = $MambaVariant
$PreflightCode = "from mamba_ssm import Mamba, Mamba2, Mamba3; cls={'mamba':Mamba,'mamba2':Mamba2,'mamba3':Mamba3}['$MambaVariant']; cls(d_model=$DModel,d_state=$DState,d_conv=4,expand=2); print('$MambaVariant preflight ok')"
& micromamba run -n $EnvName python -c $PreflightCode
if ($LASTEXITCODE -ne 0) {
    if ($NoFallback -or -not $FallbackVariant) {
        throw "Mamba variant '$MambaVariant' failed preflight. Fix dependencies or rerun without -NoFallback."
    }
    Write-Warning "Mamba variant '$MambaVariant' failed preflight; falling back to '$FallbackVariant' so the overnight run is not wasted."
    $MambaVariant = $FallbackVariant
    if ($RunName -like "*$RequestedVariant*") {
        $RunName = $RunName.Replace($RequestedVariant, $MambaVariant)
    } else {
        $RunName = "$RunName`_$MambaVariant"
    }
    $PreflightCode = "from mamba_ssm import Mamba, Mamba2; cls={'mamba':Mamba,'mamba2':Mamba2}['$MambaVariant']; cls(d_model=$DModel,d_state=$DState,d_conv=4,expand=2); print('$MambaVariant preflight ok')"
    & micromamba run -n $EnvName python -c $PreflightCode
    if ($LASTEXITCODE -ne 0) {
        throw "Fallback Mamba variant '$MambaVariant' also failed preflight."
    }
}

if (-not $ResumeFrom) {
    $Latest = Join-Path $RepoRoot "runs\$RunName\model_latest.pt"
    if (Test-Path $Latest) {
        $ResumeFrom = $Latest
        Write-Host "Auto-resume from $ResumeFrom"
    }
}

$ArgsList = @(
    "run", "-n", $EnvName,
    "python", "src\train_mamba_ppo.py",
    "--model", "mamba",
    "--mamba-variant", $MambaVariant,
    "--env-id", $EnvId,
    "--seed", "$Seed",
    "--total-steps", "$TotalSteps",
    "--num-envs", "$NumEnvs",
    "--num-steps", "$NumSteps",
    "--n-epochs", "$NEpochs",
    "--context-len", "$ContextLen",
    "--chunk-len", "$ChunkLen",
    "--batch-chunks", "$BatchChunks",
    "--d-model", "$DModel",
    "--spatial-encoder", $SpatialEncoder,
    "--spatial-layers", "2",
    "--spatial-heads", "4",
    "--mamba-layers", "$MambaLayers",
    "--d-state", "$DState",
    "--d-conv", "4",
    "--expand", "2",
    "--lr", "2.5e-4",
    "--ent-coef", "$EntCoef",
    "--ent-coef-final", "$EntCoefFinal",
    "--eval-interval", "$EvalInterval",
    "--eval-episodes", "$EvalEpisodes",
    "--save-interval", "$SaveInterval",
    "--log-interval", "$LogInterval",
    "--valid-actions", $ValidActions,
    "--run-name", $RunName
)

if ($ResumeFrom) {
    $ArgsList += @("--resume-from", $ResumeFrom)
}

Write-Host "Repo: $RepoRoot"
Write-Host "Log:  $LogPath"
Write-Host "Run:  micromamba $($ArgsList -join ' ')"

& micromamba @ArgsList 2>&1 | Tee-Object -FilePath $LogPath
