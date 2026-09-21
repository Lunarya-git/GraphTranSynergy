# setup_and_run.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Full setup + 1-epoch run for GraphTranSynergy
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File setup_and_run.ps1
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot   # directory this script lives in

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  GraphTranSynergy - Setup and 1-Epoch Run" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Create virtual environment ───────────────────────────────────────
Write-Host "[1/6] Creating virtual environment (.venv) ..." -ForegroundColor Yellow
if (-Not (Test-Path "$ROOT\.venv")) {
    python -m venv "$ROOT\.venv"
    Write-Host "      Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "      .venv already exists, skipping." -ForegroundColor Gray
}

$pip    = "$ROOT\.venv\Scripts\pip.exe"
$python = "$ROOT\.venv\Scripts\python.exe"

# ── Step 2: Upgrade pip ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/6] Upgrading pip ..." -ForegroundColor Yellow
& $pip install --upgrade pip --quiet

# ── Step 3: Install dependencies ─────────────────────────────────────────────
Write-Host ""
Write-Host "[3/6] Installing dependencies from requirements.txt ..." -ForegroundColor Yellow
Write-Host "      (This may take a few minutes on first run)" -ForegroundColor Gray

# Install torch first (CPU build) so PyG picks the right variant
& $pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu --quiet

# Install PyG base package
Write-Host "      Installing torch-geometric ..." -ForegroundColor Gray
& $pip install torch-geometric --quiet

# Install PyG optional extensions (scatter / sparse / cluster) from PyG wheel index
Write-Host "      Installing torch-scatter, torch-sparse, torch-cluster ..." -ForegroundColor Gray
& $pip install torch-scatter torch-sparse torch-cluster `
    --find-links https://data.pyg.org/whl/torch-2.13.0+cpu.html --quiet

# Install everything else
Write-Host "      Installing remaining packages ..." -ForegroundColor Gray
& $pip install rdkit networkx numpy scipy pandas scikit-learn tqdm matplotlib --quiet

Write-Host "      All packages installed." -ForegroundColor Green

# ── Step 4: Extract archives ──────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/6] Extracting data.7z and model.7z ..." -ForegroundColor Yellow
$sevenzip = "C:\Program Files\7-Zip\7z.exe"

if (-Not (Test-Path $sevenzip)) {
    Write-Host "ERROR: 7-Zip not found at $sevenzip" -ForegroundColor Red
    Write-Host "       Please install 7-Zip from https://www.7-zip.org and re-run." -ForegroundColor Red
    exit 1
}

if (-Not (Test-Path "$ROOT\data\smiles.csv")) {
    Write-Host "      Extracting data.7z ..." -ForegroundColor Gray
    & $sevenzip x "$ROOT\data.7z" -o"$ROOT" -y | Out-Null
    Write-Host "      data.7z extracted." -ForegroundColor Green
} else {
    Write-Host "      data/ already extracted, skipping." -ForegroundColor Gray
}

if (-Not (Test-Path "$ROOT\model\GraphTransformer.py")) {
    Write-Host "      Extracting model.7z ..." -ForegroundColor Gray
    & $sevenzip x "$ROOT\model.7z" -o"$ROOT\model" -y | Out-Null
    Write-Host "      model.7z extracted." -ForegroundColor Green
} else {
    Write-Host "      model/ already extracted, skipping." -ForegroundColor Gray
}

# ── Step 5: Create output directories ────────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Creating output directories ..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$ROOT\data\result\sun_pre_test\GraphTransformer" | Out-Null
Write-Host "      Output directories ready." -ForegroundColor Green

# ── Step 6: Run for one epoch ─────────────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] Running run_one_epoch.py (NUM_EPOCHS=1, 5-fold CV) ..." -ForegroundColor Yellow
Write-Host "      Output will stream below:" -ForegroundColor Gray
Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
Set-Location $ROOT
& $python "$ROOT\run_one_epoch.py"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Done! Results in data/result/sun_pre_test/GraphTransformer/" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
