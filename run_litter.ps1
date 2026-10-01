<#
.SYNOPSIS
  해안쓰레기 위치·무게 → 수거계획 파이프라인 (litter/) 실행.

.DESCRIPTION
  -Synth : 합성 데이터를 새로 만들고 전체 파이프라인 + 어블레이션 실행 (업체 데이터 없을 때 검증용)
  -Data  : 업체 데이터 폴더. 안에서 라벨/무게/텔레메트리/DSM 파일을 이름으로 찾는다.
           못 찾으면 python -m litter inspect 결과를 보고 python -m litter run 을 직접 호출.

.EXAMPLE
  .\run_litter.ps1 -Synth
  .\run_litter.ps1 -Data D:\업체데이터
#>
[CmdletBinding()]
param(
    [switch]$Synth,
    [string]$Data
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "가상환경이 없습니다. 먼저 .\setup_windows.ps1 을 실행하세요." }
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Push-Location $root
try {
    if ($Synth) {
        & $py -m litter synth --out data/litter_synth
        & $py -m litter run --images data/litter_synth/images `
            --labels data/litter_synth/labels_coco.json `
            --weights data/litter_synth/weights.csv `
            --telemetry data/litter_synth/telemetry.csv `
            --dsm data/litter_synth/dsm.tif --eval --out results/litter_synth
        # DSM 없이(2D만) — 업체 데이터가 3D 복원이 안 될 때의 성능
        & $py -m litter run --images data/litter_synth/images `
            --labels data/litter_synth/labels_coco.json `
            --weights data/litter_synth/weights.csv `
            --telemetry data/litter_synth/telemetry.csv `
            --eval --out results/litter_synth_2d
    }
    elseif ($Data) {
        & $py -m litter inspect $Data
        $find = { param($pat) Get-ChildItem -Path $Data -Recurse -File -Include $pat -ErrorAction SilentlyContinue | Select-Object -First 1 }
        $labels = & $find @("*coco*.json", "*label*.json", "*annotation*.json", "*.json")
        if (-not $labels) { throw "라벨 파일을 못 찾음 — inspect 결과를 보고 python -m litter run 을 직접 실행하세요." }
        $cmdArgs = @("-m", "litter", "run", "--labels", $labels.FullName, "--out", "results/litter_company")
        $img = Get-ChildItem -Path $Data -Recurse -File -Include *.jpg, *.jpeg, *.png | Select-Object -First 1
        if ($img) { $cmdArgs += @("--images", $img.DirectoryName) }
        $w = & $find @("*weight*.csv", "*무게*.csv")
        if ($w) { $cmdArgs += @("--weights", $w.FullName, "--eval") }
        $t = & $find @("*telemetry*.csv", "*gps*.csv", "*flight*.csv")
        if ($t) { $cmdArgs += @("--telemetry", $t.FullName) }
        $d = & $find @("*dsm*.tif")
        if ($d) { $cmdArgs += @("--dsm", $d.FullName) }
        Write-Host "실행: python $($cmdArgs -join ' ')"
        & $py @cmdArgs
    }
    else {
        Write-Host "사용법: .\run_litter.ps1 -Synth  또는  .\run_litter.ps1 -Data <업체데이터 폴더>"
    }
}
finally { Pop-Location }
