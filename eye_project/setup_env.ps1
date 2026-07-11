# Create and prepare a local Python environment for the eye-tracking project.

Write-Host "Creating virtual environment: venv"
python -m venv venv

Write-Host "Activating virtual environment"
. .\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip"
python -m pip install --upgrade pip

Write-Host "Installing project dependencies"
pip install -r requirements.txt

Write-Host "Environment ready."
