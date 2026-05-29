param(
    [ValidateSet("ti", "s")]
    [string]$Model = "ti",
    [string]$Image = "figs/examples/dogs.jpg",
    [string]$OutputDir = "local_outputs"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "EfficientSAM local demo"
Write-Host "Project: $ProjectRoot"
Write-Host "Model: $Model"
Write-Host "Image: $Image"
Write-Host "Output: $OutputDir"

$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    throw "Conda was not found in PATH. Please run this script from an Anaconda/Miniconda-enabled PowerShell."
}

conda run -n mlvr python local_demo.py --model $Model --image $Image --output-dir $OutputDir

Write-Host ""
Write-Host "Done."
Write-Host "Mask: $ProjectRoot\$OutputDir\dogs_efficientsam_${Model}_mask.png"
Write-Host "Masked image: $ProjectRoot\$OutputDir\dogs_efficientsam_${Model}_masked.png"
