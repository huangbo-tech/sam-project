# Weights

Large model files are not committed in the course-upload version of this project.

`local_demo.py` downloads the required official EfficientSAM checkpoint automatically when it is missing:

- EfficientSAM-Ti: `https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt`
- EfficientSAM-S: `https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip`

The local workspace used for testing already had the weights and successfully ran `run_project.ps1 -Model ti`.
