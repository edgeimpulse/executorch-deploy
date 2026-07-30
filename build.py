"""Edge Impulse custom deployment block: export to ExecuTorch (.pte).

Edge Impulse invokes this with:

    build.py --metadata <path to deployment-metadata.json>

The metadata file describes the impulse and points to the input and output
folders. The input folder contains the trained model in several formats plus the
Edge Impulse SDK. We package an ExecuTorch program together with a small runtime
harness into `deploy.zip` in the output folder.

Conversion path
---------------
ExecuTorch consumes PyTorch programs, while Edge Impulse provides TensorFlow /
TFLite / ONNX artifacts. This block converts the exported **ONNX** model to a
PyTorch module (via onnx2torch), then lowers it to a `.pte` with the selected
backend. If ONNX is not available or conversion fails, the block still packages
the original model plus instructions so the deployment never hard-fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile

APP_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "app")
BUILD_DIR = "/tmp/build"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExecuTorch deployment block")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--backend", type=str, default="xnnpack",
                        choices=["xnnpack", "portable"])
    args, _ = parser.parse_known_args()
    return args


def load_metadata(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def find_onnx_model(metadata: dict, input_dir: str) -> str | None:
    """Locate an ONNX model, preferring the path in the metadata."""
    for model in metadata.get("tfliteModels", []):
        onnx_path = model.get("onnxModelPath")
        if onnx_path and os.path.exists(onnx_path):
            return onnx_path
    # Fall back to a scan of the input directory.
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".onnx"):
                return os.path.join(root, name)
    return None


def convert_onnx_to_pte(onnx_path: str, backend: str, out_path: str) -> bool:
    """Convert an ONNX model to an ExecuTorch program. Returns success."""
    try:
        import onnx
        import torch
        from onnx2torch import convert
    except ImportError as exc:
        print(f"NOTE: ONNX->PyTorch tooling unavailable ({exc}); skipping .pte.")
        return False

    try:
        onnx_model = onnx.load(onnx_path)
        torch_model = convert(onnx_model).eval()

        # Derive an example input from the ONNX graph's first input shape.
        graph_input = onnx_model.graph.input[0]
        dims = graph_input.type.tensor_type.shape.dim
        shape = [(d.dim_value if d.dim_value > 0 else 1) for d in dims]
        example = (torch.randn(*shape),)

        from executorch.exir import to_edge_transform_and_lower

        exported = torch.export.export(torch_model, example)
        if backend == "xnnpack":
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )
            program = to_edge_transform_and_lower(
                exported, partitioner=[XnnpackPartitioner()]
            ).to_executorch()
        else:
            program = to_edge_transform_and_lower(exported).to_executorch()

        with open(out_path, "wb") as f:
            f.write(program.buffer)
        print(f"Wrote ExecuTorch program to {out_path} (backend={backend}).")
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue packaging
        print(f"WARNING: ExecuTorch conversion failed: {exc}")
        return False


def make_build_dir() -> None:
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)


def zip_dir(src_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, src_dir)
                zf.write(abs_path, rel_path)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)

    input_dir = metadata["folders"]["input"]
    output_dir = metadata["folders"]["output"]

    make_build_dir()

    # 1. Copy the runtime harness.
    shutil.copytree(APP_DIR, os.path.join(BUILD_DIR, "app"))

    # 2. Write the label map for the runtime.
    classes = metadata.get("classes", [])
    with open(os.path.join(BUILD_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(classes))

    # 3. Attempt ExecuTorch conversion from ONNX.
    onnx_path = find_onnx_model(metadata, input_dir)
    pte_path = os.path.join(BUILD_DIR, "model.pte")
    converted = False
    if onnx_path:
        converted = convert_onnx_to_pte(onnx_path, args.backend, pte_path)
        # Keep the source ONNX in the package as a fallback / reference.
        shutil.copy(onnx_path, os.path.join(BUILD_DIR, "model.onnx"))
    else:
        print("NOTE: no ONNX model found in the deployment inputs.")

    if not converted:
        with open(os.path.join(BUILD_DIR, "CONVERSION_NOTE.txt"), "w") as f:
            f.write(
                "ExecuTorch (.pte) was not produced automatically.\n"
                "The trained model is included as model.onnx. Convert it with:\n\n"
                "    python app/convert.py model.onnx model.pte --backend "
                f"{args.backend}\n"
            )

    # 4. Package everything into deploy.zip.
    os.makedirs(output_dir, exist_ok=True)
    deploy_zip = os.path.join(output_dir, "deploy.zip")
    zip_dir(BUILD_DIR, deploy_zip)
    print(f"Deployment package written to {deploy_zip}")


if __name__ == "__main__":
    main()
