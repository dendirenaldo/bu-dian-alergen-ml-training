<#
  capture_training.ps1 — Jalankan fine-tuning BERT sambil menyimpan
  training log lengkap (untuk bukti laporan/skripsi).

  Jalankan dari ROOT repo (Windows), misal:
    cd "D:\Dendi S3\bu-dian-alergen-ml-training"
    powershell -ExecutionPolicy Bypass -File .\artifacts\bert\capture_training.ps1

  (Jika folder repo Anda masih memakai nama lama "bert_hpc" bukan
  "artifacts\bert", atur -DataDir .\bert_hpc)
#>
param(
  [string]$Python = ".\venv\Scripts\python.exe",
  [string]$DataDir = ".\artifacts\bert",
  [string]$OutputDir = ".\artifacts\bert\results_rerun",
  [int]$Epochs = 4,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $DataDir "training_run_$stamp.log"

function W($msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $logFile -Value $line
}

# ---- Header log (bukti lingkungan & kontrak) ----
W "=== BERT FINE-TUNING EVIDENCE RUN ==="
W "Repo      : $((Get-Location).Path)"
try { W ("Git commit : " + (git rev-parse --short HEAD 2>$null)) } catch { W "Git commit : n/a" }
W "Python    : $(& $Python --version 2>&1)"
W "Command   : $Python $DataDir\train_bert_hpc.py --data-dir $DataDir\input --output-dir $OutputDir --epochs $Epochs --seed $Seed"
W "Seed      : $Seed (kontrak deterministik; hasil harus identik dengan run sebelumnya)"
W "Log file  : $logFile"
W "--- mulai pelatihan (keluaran terminal di-capture ke log) ---"

# ---- Jalankan training, semua keluaran masuk log ----
# 2>&1 | Tee-Object: tampil di layar DAN tersimpan ke file.
& $Python (Join-Path $DataDir "train_bert_hpc.py") `
  --data-dir (Join-Path $DataDir "input") `
  --output-dir $OutputDir `
  --epochs $Epochs `
  --seed $Seed 2>&1 | ForEach-Object {
    $ts = Get-Date -Format "HH:mm:ss"
    "$ts | $_"
  } | Tee-Object -FilePath $logFile -Append

$code = $LASTEXITCODE
W "--- pelatihan selesai (exit code $code) ---"
W "Ringkasan artefak di $OutputDir :"
Get-ChildItem $OutputDir -File | ForEach-Object {
  W ("  {0}  ({1:N0} bytes)" -f $_.Name, $_.Length)
}
W "LOG: $logFile"
