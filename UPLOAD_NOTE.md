# Course Upload Version

This repository is a lightweight upload package based on the locally verified EfficientSAM project.

Included:

- EfficientSAM source code
- one-click run scripts: `run_project.bat` and `run_project.ps1`
- local CUDA demo: `local_demo.py`
- sample image: `figs/examples/dogs.jpg`
- running instructions: `LOCAL_RUN.md`

Excluded:

- large checkpoint / ONNX / torchscript files
- local inference outputs
- PPT/report files

When a checkpoint is missing, `local_demo.py` downloads the official EfficientSAM checkpoint automatically.
