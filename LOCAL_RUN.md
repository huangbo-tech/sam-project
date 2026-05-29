# EfficientSAM 本地运行说明

## 环境

- Windows / PowerShell
- Conda 环境：`mlvr`
- Python：3.8.20
- PyTorch：2.0.0+cu118
- Torchvision：0.15.1+cu118
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU 8GB

## 已跑通的官方示例

在项目根目录运行：

```powershell
conda run -n mlvr python EfficientSAM_example.py
```

本机输出：

```text
Running inference using  efficientsam_ti
Running inference using  efficientsam_s
```

生成文件：

- `figs/examples/dogs_efficientsam_ti_mask.png`
- `figs/examples/dogs_efficientsam_s_mask.png`

## 本地 GPU Demo

我额外添加了 `local_demo.py`，默认使用 EfficientSAM-Ti 和官方 `dogs.jpg`，并自动选择 CUDA：

```powershell
conda run -n mlvr python local_demo.py --model ti
```

输出目录：

- `local_outputs/dogs_efficientsam_ti_mask.png`
- `local_outputs/dogs_efficientsam_ti_masked.png`

如果要跑 EfficientSAM-S：

```powershell
conda run -n mlvr python local_demo.py --model s
```

注意：`weights/efficient_sam_vits.pt` 是从官方 zip 解压得到的 100MB 以上权重文件，已加入 `.gitignore`，避免直接提交到 GitHub 普通仓库。

## 一键运行脚本

双击或在 PowerShell 中运行：

```powershell
.\run_project.bat
```

也可以直接运行 PowerShell 脚本，并指定模型：

```powershell
.\run_project.ps1 -Model ti
.\run_project.ps1 -Model s
```

默认输出仍在 `local_outputs/`。
