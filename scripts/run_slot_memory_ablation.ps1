param(
    [string]$EnvName = "mamba_env",
    [string]$EnvId = "MiniGrid-MemoryS13Random-v0",
    [string]$Seeds = "42,43,44",
    [int]$TotalSteps = 1000000,
    [int]$NumEnvs = 16,
    [int]$NumSteps = 128,
    [int]$ContextLen = 128,
    [int]$ChunkLen = 64,
    [int]$BatchChunks = 8,
    [int]$DModel = 128,
    [int]$EvalInterval = 20000,
    [int]$EvalEpisodes = 30,
    [int]$SaveInterval = 100000,
    [string]$Models = "gru,gated_attention,mamba3",
    [switch]$Curriculum,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$StructureVariants = @(
    @{
        Name = "query_pool_flatten_nomem"
        SlotExtractor = "query_pool"
        TokenMode = "flatten"
        MemoryKind = "none"
        AuxCoef = "0.0"
    },
    @{
        Name = "query_pool_fuse_nomem"
        SlotExtractor = "query_pool"
        TokenMode = "fuse"
        MemoryKind = "none"
        AuxCoef = "0.0"
    },
    @{
        Name = "iterative_fuse_nomem"
        SlotExtractor = "iterative"
        TokenMode = "fuse"
        MemoryKind = "none"
        AuxCoef = "0.0"
    },
    @{
        Name = "iterative_fuse_memory"
        SlotExtractor = "iterative"
        TokenMode = "fuse"
        MemoryKind = "episodic_cue"
        AuxCoef = "0.0"
    },
    @{
        Name = "iterative_fuse_memory_aux"
        SlotExtractor = "iterative"
        TokenMode = "fuse"
        MemoryKind = "episodic_cue"
        AuxCoef = "0.05"
    }
)

function ModelArgs([string]$ModelName) {
    if ($ModelName -eq "mamba3") {
        return @("--model", "mamba", "--mamba-variant", "mamba3", "--no-stateful-rollout")
    }
    if ($ModelName -eq "mamba2") {
        return @("--model", "mamba", "--mamba-variant", "mamba2")
    }
    if ($ModelName -eq "gated_attention") {
        return @("--model", "gated_attention", "--gated-attention-pos", "alibi")
    }
    return @("--model", $ModelName)
}

$SeedList = $Seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$ModelList = $Models.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }

foreach ($modelName in $ModelList) {
    foreach ($variant in $StructureVariants) {
        foreach ($seed in $SeedList) {
            $runName = "ablate_$($modelName)_$($variant.Name)_$($EnvId.Replace('MiniGrid-', '').Replace('-v0', ''))_seed$seed"
            $argsList = @(
                "run", "-n", $EnvName,
                "python", "src\train_mamba_ppo.py"
            )
            $argsList += ModelArgs $modelName
            $argsList += @(
                "--env-id", $EnvId,
                "--seed", $seed,
                "--total-steps", "$TotalSteps",
                "--num-envs", "$NumEnvs",
                "--num-steps", "$NumSteps",
                "--context-len", "$ContextLen",
                "--chunk-len", "$ChunkLen",
                "--batch-chunks", "$BatchChunks",
                "--d-model", "$DModel",
                "--slot-count", "4",
                "--slot-extractor", $variant.SlotExtractor,
                "--temporal-token-mode", $variant.TokenMode,
                "--memory-kind", $variant.MemoryKind,
                "--aux-recall-coef", $variant.AuxCoef,
                "--valid-actions", "0,1,2",
                "--eval-interval", "$EvalInterval",
                "--eval-episodes", "$EvalEpisodes",
                "--save-interval", "$SaveInterval",
                "--run-name", $runName
            )
            if ($Curriculum) {
                $argsList += @("--curriculum")
            }
            $commandLine = "micromamba " + ($argsList -join " ")
            Write-Host $commandLine
            if ($Execute) {
                & micromamba @argsList
                if ($LASTEXITCODE -ne 0) {
                    throw "Run failed: $commandLine"
                }
            }
        }
    }
}
