[CmdletBinding()]
param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path.TrimEnd("\")
$handoffRelative = "outputs/releases/trip-qwen3-vl-8b-system-repair-v1-rc1-final-v3"
$handoff = [System.IO.Path]::GetFullPath((Join-Path $root $handoffRelative))
$manifestPath = Join-Path $handoff "release_manifest.json"

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Verified model handoff manifest is missing: $manifestPath"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($layer in @("runtime", "adapter", "retrieval", "evidence")) {
    $record = $manifest.layers.$layer
    $archive = Join-Path $handoff $record.file
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        throw "Handoff layer is missing: $archive"
    }
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $record.sha256) {
        throw "Handoff layer hash mismatch: $layer"
    }
    if ((Get-Item -LiteralPath $archive).Length -ne $record.size_bytes) {
        throw "Handoff layer size mismatch: $layer"
    }
}

$targets = @(
    "data/yelp",
    "data/Yelp-Photos.zip",
    "data/Yelp-JSON.zip",
    "data/eval",
    "models",
    "work",
    "outputs/week5_qwen3_vl_4b",
    "outputs/week6",
    "outputs/system_repair",
    "outputs/week7",
    "outputs/week5",
    "outputs/models",
    "outputs/week4",
    "outputs/week4_qwen3_vl_4b",
    "outputs/_archives",
    "outputs/week4_qwen37_plus",
    "outputs/_logs",
    "outputs/releases/trip-qwen3-vl-8b-system-repair-v1-rc1",
    "outputs/releases/trip-qwen3-vl-8b-system-repair-v1-rc1-final",
    "outputs/releases/trip-qwen3-vl-8b-system-repair-v1-rc1-final-v2",
    "secrets/chrome-plugin-repair-backup-20260813"
)

$rootPrefix = $root + "\"
$resolvedTargets = @()
foreach ($relative in $targets) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $candidate.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escaped repository: $candidate"
    }
    if ($candidate.Equals($handoff, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target includes the model handoff package"
    }
    if (Test-Path -LiteralPath $candidate) {
        $resolvedTargets += $candidate
    }
}

$totalBytes = 0L
foreach ($target in $resolvedTargets) {
    $item = Get-Item -LiteralPath $target -Force
    if ($item.PSIsContainer) {
        $size = (Get-ChildItem -LiteralPath $target -File -Recurse -Force -ErrorAction Stop |
            Measure-Object Length -Sum).Sum
        if ($null -eq $size) {
            $size = 0L
        }
    }
    else {
        $size = $item.Length
    }
    $totalBytes += [long]$size
    Write-Output ("{0}`t{1}" -f $size, $target)
}

if (-not $Apply) {
    Write-Output ("DRY_RUN total_bytes={0} targets={1}" -f $totalBytes, $resolvedTargets.Count)
    exit 0
}

foreach ($target in $resolvedTargets) {
    $item = Get-Item -LiteralPath $target -Force
    if ($item.PSIsContainer) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    else {
        Remove-Item -LiteralPath $target -Force
    }
}

Write-Output ("CLEANED total_bytes={0} targets={1}" -f $totalBytes, $resolvedTargets.Count)
