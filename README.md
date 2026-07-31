# ExecuTorch deployment block

A custom [Edge Impulse deployment block](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-deployment-blocks) (enterprise) that packages a trained impulse into an [ExecuTorch](https://pytorch.org/executorch/) program (`.pte`) with a small Python runtime harness.

Because ExecuTorch consumes PyTorch programs while Edge Impulse emits TensorFlow/TFLite/ONNX, the block converts the exported **ONNX** model to PyTorch (via `onnx2torch`) and lowers it to a `.pte` with the chosen backend. If conversion tooling is unavailable, it still packages `model.onnx` plus a conversion note so a deploy never hard-fails.

# Compatible Learn Blocks Classification, Timeseries, Object Detection FOMO
| Name | link |
|------|---------|
| Timeseries | https://github.com/edgeimpulse/executorch-pytorch-timeseries-block |
| Classification | https://github.com/edgeimpulse/executorch-pytorch-classification-block |
| FOMO | https://github.com/edgeimpulse/executorch-pytorch-object-detection-fomo-block |

# Deploy - Run on Android :
https://github.com/edgeimpulse/executorch-android-app

## Files

| File | Purpose |
|------|---------|
| `parameters.json` | Deploy block metadata + the backend selector shown in Studio. |
| `build.py` | Entrypoint. Reads `deployment-metadata.json`, converts, and writes `deploy.zip`. |
| `app/run_pte.py` | Runtime harness: loads a `.pte` and runs one forward pass. |
| `app/convert.py` | Offline ONNX → `.pte` converter (same logic as the block). |
| `Dockerfile` | Builds the block container. |
| `requirements.txt` | Pinned dependencies (`onnx`, `onnx2torch`, `torch`, `executorch`). |

## Block interface

- **Input** (`--metadata`): `deployment-metadata.json`, which points to `folders.input` (SDK + trained model in several formats) and `folders.output`.
- **Parameter**: `--backend` (`xnnpack` or `portable`).
- **Output**: `deploy.zip` in `folders.output`, containing `model.pte` (when converted), `model.onnx`, `labels.txt`, and the `app/` harness.

## Test locally

```bash
# Download the deployment inputs for a project into ./input
edge-impulse-blocks runner --download-data input/

docker build -t executorch-deploy .
docker run --rm -v "$PWD":/home executorch-deploy \
    --metadata /home/input/deployment-metadata.json
```

The resulting `deploy.zip` is written to the output directory reported in the metadata.

## Push to Edge Impulse

```bash
edge-impulse-blocks init   # first time only; choose "Deployment block"
edge-impulse-blocks push
```

It then appears as a **Custom block** option on the project Deployment page.

## Status

This is an MVP. The ONNX → PyTorch → ExecuTorch path works for common CNN graphs; exotic ops may need a custom partitioner or manual conversion via `app/convert.py`. Validate on your target model before relying on it in production.
