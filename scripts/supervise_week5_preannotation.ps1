param(
    [Parameter(Mandatory = $true)]
    [string]$SshHost,
    [string]$SshUser = "ecs-user",
    [string]$IdentityFile = "secrets/trip-project-key.pem",
    [string]$RunId = "week5_full_preannotation_qwen3_vl_4b_20260809_b",
    [string]$Config = "configs/week5_dataset_qwen3_vl_4b_gpu.json",
    [int]$LocalPort = 18001,
    [int]$RemotePort = 8001,
    [int]$RestartDelaySeconds = 15
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$identityPath = (Resolve-Path -LiteralPath $IdentityFile).Path
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$sshExe = (Get-Command ssh -ErrorAction Stop).Source
$runDir = Join-Path $repoRoot "outputs/week5_qwen3_vl_4b/runs/$RunId"
$logDir = Join-Path $runDir "supervisor"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusPath = Join-Path $logDir "status.json"
$donePath = Join-Path $logDir "finished.json"
$supervisorLog = Join-Path $logDir "supervisor.log"

$mutex = New-Object System.Threading.Mutex($false, "Global\TripProjectWeek5Preannotation")
if (-not $mutex.WaitOne(0, $false)) {
    throw "another Week 5 preannotation supervisor is already running"
}

function Write-SupervisorLog([string]$Message) {
    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -Encoding UTF8 -LiteralPath $supervisorLog -Value "$timestamp $Message"
}

function Write-Status([string]$State, [int]$TunnelPid = 0, [int]$RunnerPid = 0) {
    @{
        state = $State
        run_id = $RunId
        tunnel_pid = $TunnelPid
        runner_pid = $RunnerPid
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $statusPath
}

function Test-ModelEndpoint {
    try {
        $response = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$LocalPort/v1/models" `
            -Method Get `
            -TimeoutSec 5
        return $null -ne $response.data
    }
    catch {
        return $false
    }
}

function Start-SshTunnel {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logDir "ssh_${timestamp}.stdout.log"
    $stderr = Join-Path $logDir "ssh_${timestamp}.stderr.log"
    $arguments = @(
        "-N",
        "-L", "${LocalPort}:127.0.0.1:${RemotePort}",
        "-i", $identityPath,
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ConnectTimeout=15",
        "${SshUser}@${SshHost}"
    )
    return Start-Process `
        -FilePath $sshExe `
        -ArgumentList $arguments `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr
}

function Wait-ForTunnel([System.Diagnostics.Process]$Tunnel) {
    for ($attempt = 1; $attempt -le 24; $attempt++) {
        if ($Tunnel.HasExited) {
            return $false
        }
        if (Test-ModelEndpoint) {
            return $true
        }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Start-Runner([bool]$RetryFailures) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logDir "runner_${timestamp}.stdout.log"
    $stderr = Join-Path $logDir "runner_${timestamp}.stderr.log"
    $arguments = @(
        "scripts/manage_week5_dataset.py",
        "--config", $Config,
        "preannotate-all",
        "--run-id", $RunId,
        "--resume"
    )
    if ($RetryFailures) {
        $arguments += "--retry-failures"
    }
    return Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $arguments `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr
}

try {
    if (Test-Path -LiteralPath $donePath) {
        Write-Status "already_finished"
        exit 0
    }
    $cleanupPass = $false
    while ($true) {
        Write-Status "starting_tunnel"
        $tunnel = Start-SshTunnel
        if (-not (Wait-ForTunnel $tunnel)) {
            Write-SupervisorLog "SSH tunnel failed health check; retrying"
            if (-not $tunnel.HasExited) {
                Stop-Process -Id $tunnel.Id -Force
            }
            Write-Status "waiting_for_tunnel"
            Start-Sleep -Seconds $RestartDelaySeconds
            continue
        }

        Write-SupervisorLog "SSH tunnel healthy; starting runner cleanup=$cleanupPass"
        $runner = Start-Runner $cleanupPass
        Write-Status "running" $tunnel.Id $runner.Id
        $runner.WaitForExit()
        $exitCode = $runner.ExitCode
        Write-SupervisorLog "runner exited code=$exitCode cleanup=$cleanupPass"

        if (-not $tunnel.HasExited) {
            Stop-Process -Id $tunnel.Id -Force
        }
        if ($exitCode -ne 0) {
            Write-Status "runner_interrupted"
            Start-Sleep -Seconds $RestartDelaySeconds
            continue
        }

        $summaryPath = Join-Path $runDir "summary.json"
        if (-not (Test-Path -LiteralPath $summaryPath)) {
            Write-SupervisorLog "runner returned zero without summary; restarting"
            Start-Sleep -Seconds $RestartDelaySeconds
            continue
        }
        $summary = Get-Content -Raw -Encoding UTF8 -LiteralPath $summaryPath | ConvertFrom-Json
        if ($summary.status -eq "completed") {
            $summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -LiteralPath $donePath
            Write-Status "completed"
            break
        }
        if (-not $cleanupPass) {
            $cleanupPass = $true
            Write-SupervisorLog "main pass complete with failures; starting one cleanup pass"
            continue
        }

        @{
            status = "partial_after_cleanup"
            summary = $summary
            updated_at = (Get-Date).ToUniversalTime().ToString("o")
        } | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -LiteralPath $donePath
        Write-Status "partial_after_cleanup"
        break
    }
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
