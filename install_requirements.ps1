python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    $pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ([double]$pyVersion -ge 3.13) {
        Write-Host "Dependencies installed."
        Write-Host "Note: TensorFlow is skipped on Python $pyVersion."
        Write-Host "Use Python 3.10-3.12 to train and run the LSTM model."
    }
}
