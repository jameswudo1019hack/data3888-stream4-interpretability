"""Model loading, preprocessing, prediction and Grad-CAM for the Shiny app.

Matches the exact training preprocess used in scripts/stream4_pipeline.py:
PIL image -> resize to 100x100 -> /255 -> ImageNet normalize ->
bilinear interpolate to 224x224 (the size the model actually sees).

Both seed-42 checkpoints are loaded once at startup and reused.
"""

from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from pytorch_grad_cam import GradCAM

ROOT = Path(__file__).resolve().parent.parent

CLASSES = ["Immune", "Stromal", "Tumour"]   # alphabetical, matches sorted(GROUPS.keys())
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOAD_SIZE = 100        # native patch size used at training load
INFERENCE_SIZE = 224   # the size the model sees after on-device resize
SEED = 42              # representative checkpoint

CHECKPOINTS = {
    "resnet50": (
        ROOT / "figures" / "v5_grouped" / "resnet50" / f"seed_{SEED}" / "resnet50_stream4.pt"
    ),
    "vit_small_patch16_224": (
        ROOT / "figures" / "v5_grouped" / "vit_small_patch16_224"
        / f"seed_{SEED}" / "vit_small_patch16_224_stream4.pt"
    ),
}

DISPLAY_NAME = {
    "resnet50": "ResNet-50",
    "vit_small_patch16_224": "ViT-small",
}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()


def _side_for_area(area_frac: float, side: int) -> int:
    """Centred-square side length (px) corresponding to `area_frac` of the image area."""
    return int(round(np.sqrt(max(0.0, min(1.0, area_frac))) * side))


def _fill_centre(arr_hw3: np.ndarray, area_frac: float, fill: np.ndarray) -> np.ndarray:
    """Replace a centred square of given area fraction with `fill` (uint8 RGB)."""
    if area_frac <= 0:
        return arr_hw3
    H, W = arr_hw3.shape[:2]
    s = _side_for_area(area_frac, H)
    if s == 0:
        return arr_hw3
    y0 = (H - s) // 2; x0 = (W - s) // 2
    out = arr_hw3.copy()
    out[y0:y0 + s, x0:x0 + s] = fill
    return out


def _fill_edge(arr_hw3: np.ndarray, area_frac: float, fill: np.ndarray) -> np.ndarray:
    """Replace everything *outside* a centred kept-square with `fill`,
    such that the filled (edge) area covers `area_frac`."""
    if area_frac <= 0:
        return arr_hw3
    H, W = arr_hw3.shape[:2]
    keep = 1.0 - area_frac
    s = _side_for_area(keep, H)
    out = np.full_like(arr_hw3, fill)
    if s > 0:
        y0 = (H - s) // 2; x0 = (W - s) // 2
        out[y0:y0 + s, x0:x0 + s] = arr_hw3[y0:y0 + s, x0:x0 + s]
    return out


def preprocess(
    pil_img: Image.Image,
    blur_sigma: float = 0.0,
    mask_type: str = "none",
    area_frac: float = 0.0,
) -> torch.Tensor:
    """PIL -> normalized (1,3,224,224) tensor on DEVICE, ready for both model and Grad-CAM.

    Perturbation order (when stacked): centre/edge fill FIRST, then blur.
    Rationale: the fill is the "structural" perturbation (region replaced
    with H&E pink), and blur is a "quality" perturbation that should blend
    the fill boundary the same way it would blend any other edge in the
    image. Doing it the other way (blur then fill) leaves a sharp fill
    region pasted on a blurred background, which looks unnatural.

    `blur_sigma > 0`: Gaussian blur radius at native 100x100 (matches v6.0).
    `mask_type` in {"none", "centre", "edge"} and `area_frac` in [0, 1]:
        match the v6.1 study. H&E pink = per-image RGB mean (single-image
        analogue of the v6.1 per-test-set mean).
    """
    img = pil_img.convert("RGB").resize((LOAD_SIZE, LOAD_SIZE))
    arr = np.asarray(img, dtype=np.uint8)

    if mask_type in ("centre", "edge") and area_frac > 0:
        pink = arr.mean(axis=(0, 1)).astype(np.uint8)  # per-image H&E mean
        if mask_type == "centre":
            arr = _fill_centre(arr, area_frac, pink)
        else:
            arr = _fill_edge(arr, area_frac, pink)

    if blur_sigma > 0:
        arr = np.asarray(
            Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=float(blur_sigma))),
            dtype=np.uint8,
        )

    arr_f = arr.astype(np.float32) / 255.0
    arr_f = (arr_f - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr_f.transpose(2, 0, 1)).unsqueeze(0)
    t = F.interpolate(t, size=INFERENCE_SIZE, mode="bilinear", align_corners=False)
    return t.contiguous().to(DEVICE)


def denorm_for_display(x_224: torch.Tensor) -> np.ndarray:
    """(1,3,224,224) normalized tensor -> (224,224,3) float array in [0,1] for plt.imshow."""
    arr = x_224[0].detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * IMAGENET_STD + IMAGENET_MEAN, 0, 1).astype(np.float32)


def _vit_reshape_transform(tensor, h: int = 14, w: int = 14):
    # Drop the CLS token, fold the 196 patch tokens back into a (14,14) grid.
    return (
        tensor[:, 1:, :]
        .reshape(tensor.size(0), h, w, tensor.size(-1))
        .permute(0, 3, 1, 2)
    )


class ModelBundle:
    """A loaded model plus its Grad-CAM extractor."""

    def __init__(self, name: str):
        self.name = name
        ckpt_path = CHECKPOINTS[name]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        # weights_only=False: our checkpoint dict carries the class list + config,
        # not just the state_dict. Trusted because we wrote it ourselves.
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        self.model = timm.create_model(name, pretrained=False, num_classes=len(CLASSES))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(DEVICE).eval()

        if name.startswith("vit"):
            self.cam = GradCAM(
                model=self.model,
                target_layers=[self.model.blocks[-1].norm1],
                reshape_transform=_vit_reshape_transform,
            )
        elif name.startswith("resnet"):
            self.cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
        else:
            raise ValueError(f"Grad-CAM target layer not configured for {name}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        logits = self.model(x)
        return F.softmax(logits, dim=1)[0].cpu().numpy()

    def gradcam(self, x: torch.Tensor) -> np.ndarray:
        # targets=None -> CAM for the predicted class
        return self.cam(input_tensor=x.contiguous(), targets=None)[0]


_BUNDLES: dict[str, ModelBundle] = {}


def get_bundle(name: str) -> ModelBundle:
    if name not in _BUNDLES:
        _BUNDLES[name] = ModelBundle(name)
    return _BUNDLES[name]


def infer_both(
    pil_img: Image.Image,
    blur_sigma: float = 0.0,
    mask_type: str = "none",
    area_frac: float = 0.0,
):
    """Run preprocess + both models + Grad-CAM. Returns:

    rgb_224 : np.ndarray (224,224,3) float [0,1] -- the image the model actually sees
    results : dict[name] -> {"probs": (3,) np.float32, "cam": (224,224) float [0,1]}
    """
    x = preprocess(pil_img, blur_sigma=blur_sigma,
                   mask_type=mask_type, area_frac=area_frac)
    rgb = denorm_for_display(x)
    results = {}
    for name in ("resnet50", "vit_small_patch16_224"):
        bundle = get_bundle(name)
        results[name] = {
            "probs": bundle.predict(x),
            "cam": bundle.gradcam(x),
        }
    return rgb, results


def warm_up() -> None:
    """Force-load both checkpoints at app startup (so the first user click is fast)."""
    for name in CHECKPOINTS:
        get_bundle(name)
