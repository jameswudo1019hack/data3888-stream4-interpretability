"""Stream 4 — interpretability demo (Shiny for Python).

Two tabs:
  - Demo:     upload / pick an H&E patch, see ResNet50 vs ViT side-by-side
              with Grad-CAM heatmaps.
  - Analysis: the v6.1 centre-vs-edge mask study and baseline performance
              that motivate this whole investigation.

Run from the project root:

    shiny run --reload app/app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shinyswatch
from matplotlib.colors import to_rgba
from PIL import Image
from shiny import App, reactive, render, ui

from inference import CLASSES, infer_both, warm_up

# -----------------------------------------------------------------------
# Paths + static asset mounts
# -----------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
ASSETS_DIR = APP_DIR / "assets"
DEMO_DIR = APP_DIR / "demo_images"
FIGURES_DIR = PROJECT_ROOT / "figures"

# -----------------------------------------------------------------------
# Layout config
# -----------------------------------------------------------------------

CLASS_COLOURS = {
    "Tumour":  "#e63946",
    "Immune":  "#1d72ad",
    "Stromal": "#2a9d8f",
}

CATEGORY_LABELS = {
    "easy":            "Easy — both models high-confidence correct",
    "disagreement":    "Disagreement — models predict different classes",
    "low_confidence":  "Low confidence — both models uncertain",
}

CLASS_SHORT = {"tum": "Tumour", "imm": "Immune", "str": "Stromal"}


plt.rcParams.update({
    "font.family":      ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font.size":        11,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#444444",
    "xtick.color":      "#666666",
    "ytick.color":      "#333333",
    "savefig.facecolor": "white",
    "figure.facecolor":  "white",
})


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _pretty_demo_label(stem: str) -> str:
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return stem
    cls = CLASS_SHORT.get(parts[1], parts[1].title())
    rest = parts[2] if len(parts) > 2 else ""
    rest = rest.replace("plus", "+")
    if rest.startswith("R-") and "_V-" in rest:
        r, _, v = rest.partition("_V-")
        return f"{cls} · ResNet→{_expand(r[2:])} / ViT→{_expand(v)}"
    if rest == "uncertain":
        return f"{cls} · uncertain"
    return f"{cls} · {rest}"


def _expand(short_cls: str) -> str:
    return {"Tum": "Tumour", "Imm": "Immune", "Str": "Stromal"}.get(short_cls, short_cls)


def _build_demo_choices() -> dict:
    choices: dict[str, dict[str, str]] = {"—": {"none": "(no demo selected)"}}
    for subdir, group_label in CATEGORY_LABELS.items():
        folder = DEMO_DIR / subdir
        if not folder.exists():
            continue
        group = {}
        for p in sorted(folder.glob("*.png")):
            rel = p.relative_to(APP_DIR).as_posix()
            group[rel] = _pretty_demo_label(p.stem)
        if group:
            choices[group_label] = group
    return choices


def _fade(hex_colour: str, alpha: float = 0.18):
    return to_rgba(hex_colour, alpha)


def _figure_img(src: str, alt: str = "", max_width: str = "920px"):
    return ui.tags.img(
        src=src,
        alt=alt,
        style=(
            f"width: 100%; max-width: {max_width}; "
            "display: block; margin: 0.5rem auto; "
            "border-radius: 4px;"
        ),
    )


# -----------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------

custom_css = """
/* Navbar brand */
.navbar-brand img {
    height: 38px;
    vertical-align: middle;
    margin-right: 12px;
    /* Source SVG is dark-on-transparent; invert so the wordmark reads white
       on the flatly navbar. */
    filter: brightness(0) invert(1);
}
.navbar-brand .brand-text {
    font-weight: 500; font-size: 1.05rem; vertical-align: middle;
    color: #ffffff;
}

/* Page hero (Demo tab) */
.app-hero {
    padding: 1.25rem 1.5rem 0.75rem;
    border-bottom: 1px solid #eee;
    background: linear-gradient(to right, #fafbfc, #ffffff);
    margin: -1rem -1rem 1rem -1rem;
}
.app-hero h2 { margin: 0; font-weight: 600; color: #2c3e50; }
.app-hero .sub { color: #6c757d; font-size: 0.95rem; margin-top: 0.25rem; }

/* Analysis hero */
.analysis-hero {
    padding: 1.5rem 1rem 0.5rem 1rem;
    margin-bottom: 0.5rem;
}
.analysis-hero h2 { color: #2c3e50; margin-bottom: 0.25rem; font-weight: 600; }
.analysis-hero .sub { color: #6c757d; font-size: 1rem; }

/* Cards */
.card { box-shadow: 0 1px 3px rgba(0,0,0,0.04); border: 1px solid #e9ecef; }
.card-header {
    background-color: #ffffff;
    border-bottom: 1px solid #f0f0f0;
    font-weight: 600;
    color: #2c3e50;
}
.card-footer { background-color: #fafbfc; font-size: 0.85rem; color: #6c757d; }

/* Verdict alert */
.verdict-alert {
    padding: 1rem 1.25rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    font-size: 1rem;
}
.verdict-agree    { background-color: #e8f6f0; color: #1f6f4d; border-left: 4px solid #2a9d8f; }
.verdict-disagree { background-color: #fff7e6; color: #8a5a00; border-left: 4px solid #f4a261; }
.verdict-empty    { background-color: #f7f7f9; color: #6c757d; border-left: 4px solid #ced4da; }

.class-pill {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #fff;
}

.takeaway {
    border-left: 3px solid #2a9d8f;
    padding: 0.4rem 0.9rem;
    background: #f4faf8;
    color: #1f3a3a;
    margin: 0.6rem 0 0.4rem;
    border-radius: 0 4px 4px 0;
}
.takeaway b { color: #1f6f4d; }

.recalculating { opacity: 0.55; transition: opacity 0.15s ease-in-out; }
"""


# -----------------------------------------------------------------------
# DEMO TAB
# -----------------------------------------------------------------------

demo_sidebar = ui.sidebar(
    ui.h5("Try an image", style="margin-top: 0;"),
    ui.input_file(
        "upload",
        "Upload an H&E patch",
        accept=[".png", ".jpg", ".jpeg"],
        multiple=False,
        button_label="Browse…",
        placeholder="No file chosen",
    ),
    ui.input_select(
        "demo_image",
        "Or pick a curated demo:",
        choices=_build_demo_choices(),
        selected="none",
    ),
    ui.hr(),
    ui.h6("Degrade the input", style="margin-bottom: 0.25rem;"),
    ui.input_slider(
        "blur_sigma",
        "Gaussian blur σ",
        min=0, max=10, value=0, step=0.5,
        ticks=True,
    ),
    ui.tags.div(
        "Higher σ = stronger blur. Applied at native 100×100 before the model upsamples to 224×224.",
        style="font-size: 0.8rem; color: #6c757d; margin-top: -0.25rem;",
    ),
    ui.input_radio_buttons(
        "mask_type",
        "Mask type (H&E-pink fill)",
        choices={"none": "None", "centre": "Centre", "edge": "Edge"},
        selected="none",
        inline=True,
    ),
    ui.input_slider(
        "area_frac",
        "Fill area fraction",
        min=0.0, max=0.7, value=0.0, step=0.05,
        ticks=True,
    ),
    ui.tags.div(
        "Fill applied first, blur second — so combinations look natural.",
        style="font-size: 0.8rem; color: #6c757d; margin-top: -0.25rem;",
    ),
    ui.hr(),
    ui.h6("About", style="margin-bottom: 0.5rem;"),
    ui.markdown(
        "Both models were fine-tuned on **1000 patches per class** "
        "(Tumour / Immune / Stromal) from breast-cancer H&E. "
        "Shown here: the **seed-42** checkpoints. "
        "Grad-CAM highlights pixels that most increased the predicted-class "
        "confidence."
    ),
    width=320,
    open="open",
)

demo_main = ui.div(
    ui.tags.div(
        ui.tags.h2("Where do the models look?"),
        ui.tags.div(
            "Upload a cell patch (or pick one of the curated demos) and watch "
            "both models classify it, with Grad-CAM showing what each one focused on.",
            class_="sub",
        ),
        class_="app-hero",
    ),
    ui.output_ui("verdict"),
    ui.layout_columns(
        ui.card(
            ui.card_header("Input image"),
            ui.output_plot("input_preview", height="280px"),
            ui.card_footer("224×224 after resize + ImageNet normalisation"),
        ),
        ui.card(
            ui.card_header("ResNet-50 (CNN)"),
            ui.output_plot("resnet_probs", height="180px"),
            ui.output_plot("resnet_cam",   height="280px"),
        ),
        ui.card(
            ui.card_header("ViT-small (Transformer)"),
            ui.output_plot("vit_probs", height="180px"),
            ui.output_plot("vit_cam",   height="280px"),
        ),
        col_widths=(4, 4, 4),
        gap="1rem",
    ),
    ui.card(
        ui.card_header("How to read this"),
        ui.markdown(
            "- **Bars** show softmax probability per class. The predicted "
            "class is in solid colour; the others are faded.\n"
            "- **Grad-CAM heatmap** highlights pixels that most increased the "
            "model's confidence in its predicted class. Red = high attention, "
            "blue = low. Computed at the last conv layer (ResNet) / last "
            "attention block (ViT).\n"
            "- **Expected pattern** (from the v6.1 mask study on the "
            "**Analysis** tab): ResNet-50 puts a sharp red blob on the centre "
            "cell; ViT-small spreads attention across the patch. Filling 5 % "
            "of the centre with H&E-pink collapses ResNet's Immune-class "
            "accuracy from 0.64 → ~0; ViT is largely unaffected."
        ),
    ),
)


# -----------------------------------------------------------------------
# ANALYSIS TAB
# -----------------------------------------------------------------------

analysis_main = ui.div(
    ui.tags.div(
        ui.tags.h2("Where do CNNs and Transformers look on H&E?"),
        ui.tags.div(
            "Comparing ResNet-50 vs ViT-small on breast-cancer cell-identity "
            "classification — full methodology, headline finding, per-class "
            "breakdown, and baseline performance.",
            class_="sub",
        ),
        class_="analysis-hero",
    ),

    # --- Research question ---------------------------------------------
    ui.card(
        ui.card_header("Research question"),
        ui.markdown(
            "When a CNN and a Vision Transformer both classify the same H&E "
            "patch into **Tumour / Immune / Stromal**, do they rely on the same "
            "image regions?\n\n"
            "We test this by **selectively hiding** parts of the image at test "
            "time — first the central cell, then the surrounding tissue — and "
            "measuring how each architecture's accuracy responds. If both rely on "
            "the centre cell, hiding it should crash both. If ViT genuinely uses "
            "context, hiding the edges should hurt ViT but not ResNet."
        ),
    ),

    # --- Headline figure ------------------------------------------------
    ui.card(
        ui.card_header("Main finding — accuracy under centre vs edge fill"),
        _figure_img("/figures/v5_grouped/noise_v2/combined.png",
                    alt="Combined centre + edge mask sweep"),
        ui.markdown(
            "**Solid line = centre fill** (central area replaced with H&E-pink). "
            "**Dashed line = edge fill** (surrounding area replaced). "
            "x-axis = fraction of total pixel area filled — same scale for both."
        ),
        ui.tags.div(
            ui.markdown(
                "**ResNet stores its decision signal at the centre.** Filling "
                "**5 % of the centre area** with the local mean colour drops accuracy "
                "0.69 → 0.51. The **same fraction at the edges** has zero effect.\n\n"
                "**ViT spreads its decision across the patch.** Neither centre nor "
                "edge fill alone breaks it; both produce a gradual, comparable decline."
            ),
            class_="takeaway",
        ),
    ),

    # --- Per-class -----------------------------------------------------
    ui.card(
        ui.card_header("Per-class breakdown — F1 by cell type"),
        _figure_img("/figures/v5_grouped/noise_v2/per_class.png",
                    alt="Per-class F1 across the centre vs edge sweep"),
        ui.markdown(
            "The **Immune** class is the clearest evidence of ResNet's "
            "centre-dependence: F1 collapses from 0.64 → ~0 at just 5 % "
            "centre fill (solid blue line), while edge fill at the same area "
            "(dashed blue line) leaves it untouched.\n\n"
            "ViT (bottom row) shows no such asymmetry on any class. Stromal "
            "F1 for ViT is the most stable line in the figure."
        ),
    ),

    # --- Methodology — visual ------------------------------------------
    ui.card(
        ui.card_header("What the perturbations look like"),
        _figure_img("/figures/v5_grouped/noise_v2/fill_examples.png",
                    alt="Visual reference of the fill scheme"),
        ui.markdown(
            "**Top row:** centre fill grows outward as area increases. "
            "**Bottom row:** edge fill shrinks the kept centre as area increases. "
            "Both use the local **H&E-pink mean colour** rather than black to "
            "stay in-distribution — pure black was the OOD confound in our "
            "v6.0 pass."
        ),
    ),

    # --- Baseline performance ------------------------------------------
    ui.card(
        ui.card_header("Baseline performance — confusion matrices"),
        ui.layout_columns(
            ui.div(
                _figure_img("/figures/v5_grouped/resnet50/seed_42/confusion_matrix.png",
                            alt="ResNet50 confusion matrix",
                            max_width="100%"),
                ui.tags.div("ResNet-50 — seed 42",
                            style="text-align: center; font-size: 0.9rem; color: #555;"),
            ),
            ui.div(
                _figure_img("/figures/v5_grouped/vit_small_patch16_224/seed_42/confusion_matrix.png",
                            alt="ViT-small confusion matrix",
                            max_width="100%"),
                ui.tags.div("ViT-small — seed 42",
                            style="text-align: center; font-size: 0.9rem; color: #555;"),
            ),
            col_widths=(6, 6),
            gap="1rem",
        ),
        ui.markdown(
            "Both models reach **~0.70 test accuracy** on the held-out 600 "
            "patches (200/class). The hardest class for both is **Immune** — "
            "fine-grained immune cell types are easy to confuse with Stromal."
        ),
    ),

    # --- Methodology summary -------------------------------------------
    ui.card(
        ui.card_header("How we got here — methodology"),
        ui.markdown(
            "- **Data**: 5 425 H&E cell patches drawn from Xenium breast-cancer "
            "imaging, mapped down to **3 broad classes** (Tumour / Immune / "
            "Stromal) per the Week 8 tutor directive. Native patch size 100×100.\n"
            "- **Sampling**: 1000 patches per class per seed (balanced), drawn "
            "with `np.random.default_rng(SEED).choice`.\n"
            "- **Splits**: stratified 70/10/20 train/val/test (sklearn).\n"
            "- **Models**: `timm` ImageNet-pretrained — **ResNet-50** and "
            "**ViT-small-patch16-224** — fine-tuned end-to-end at LR 3e-5, "
            "batch 32, 10 epochs with early stopping on val accuracy.\n"
            "- **Augmentation**: random h/v flip + 90° rotation. No "
            "class-weighting (balanced sampling already done at load time).\n"
            "- **Repeats**: 5 seeds (42–46) per architecture = 10 fully "
            "independent (model, test-set) pairs. Each bar / line in the "
            "results above is mean ± 1 std across those 5 seeds.\n"
            "- **Grad-CAM**: ResNet — last residual block (`layer4[-1]`); "
            "ViT — norm before the last attention block (`blocks[-1].norm1`) "
            "with the CLS token dropped and patch tokens folded to a 14×14 grid."
        ),
    ),

    # --- Verdict --------------------------------------------------------
    ui.card(
        ui.card_header("Verdict on the tutor's hypothesis"),
        ui.markdown(
            "> *\"ResNet uses centre, ViT uses edges.\"* — Week 9 tutor framing.\n\n"
            "- **ResNet uses centre** → ✓ strongly confirmed. The ~30-point "
            "asymmetry between centre-fill and edge-fill at small area "
            "fractions is unambiguous.\n"
            "- **ViT uses edges** → partially confirmed. ViT is broadly more "
            "robust than ResNet to either fill location, and *very slightly* "
            "prefers having the centre. It is nowhere near as edge-dependent "
            "as ResNet is centre-dependent."
        ),
        ui.tags.div(
            ui.markdown(
                "**Cleaner phrasing:** ResNet localises its decisions at the "
                "centre cell; ViT distributes its decisions across the patch."
            ),
            class_="takeaway",
        ),
    ),
)


# -----------------------------------------------------------------------
# Full UI — page_navbar with two tabs
# -----------------------------------------------------------------------

navbar_brand = ui.tags.span(
    ui.tags.img(src="/assets/usyd_logo.svg", alt="USyd"),
    ui.tags.span("Stream 4 — Interpretability Demo", class_="brand-text"),
    class_="navbar-brand",
    style="display: inline-flex; align-items: center;",
)


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Demo",
        ui.layout_sidebar(demo_sidebar, demo_main),
    ),
    ui.nav_panel(
        "Analysis",
        analysis_main,
    ),
    ui.head_content(ui.tags.style(custom_css)),
    title=navbar_brand,
    theme=shinyswatch.theme.flatly,
    fillable=False,
    id="main_nav",
)


# -----------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------

def server(input, output, session):
    warm_up()

    @reactive.calc
    def current_pil() -> Image.Image | None:
        uploaded = input.upload()
        if uploaded:
            return Image.open(uploaded[0]["datapath"]).convert("RGB")
        choice = input.demo_image()
        if choice and choice != "none":
            return Image.open(APP_DIR / choice).convert("RGB")
        return None

    @reactive.calc
    def inference():
        pil = current_pil()
        if pil is None:
            return None
        rgb, results = infer_both(
            pil,
            blur_sigma=float(input.blur_sigma()),
            mask_type=input.mask_type(),
            area_frac=float(input.area_frac()),
        )
        return {"rgb": rgb, "results": results}

    @render.ui
    def verdict():
        result = inference()
        if result is None:
            return ui.div(
                "Upload an image or pick a demo to start.",
                class_="verdict-alert verdict-empty",
            )

        r_probs = result["results"]["resnet50"]["probs"]
        v_probs = result["results"]["vit_small_patch16_224"]["probs"]
        r_pred = CLASSES[int(r_probs.argmax())]
        v_pred = CLASSES[int(v_probs.argmax())]
        r_conf = float(r_probs.max())
        v_conf = float(v_probs.max())
        blur = float(input.blur_sigma())
        mask_type = input.mask_type()
        area = float(input.area_frac())

        def pill(cls: str):
            return ui.tags.span(cls, class_="class-pill",
                                style=f"background-color: {CLASS_COLOURS[cls]};")

        mods = []
        if mask_type in ("centre", "edge") and area > 0:
            mods.append(f"{mask_type} fill {area:.0%}")
        if blur > 0:
            mods.append(f"σ={blur:.1f} blur")
        mods_tag = (
            ui.tags.span(
                " · " + ", ".join(mods),
                style="margin-left: 0.5rem; color: #555; font-style: italic;",
            )
            if mods else ""
        )

        if r_pred == v_pred:
            return ui.div(
                ui.tags.strong("Both models agree: "),
                pill(r_pred),
                ui.tags.span(
                    f"  ·  ResNet {r_conf:.0%}  ·  ViT {v_conf:.0%}",
                    style="margin-left: 0.5rem;",
                ),
                mods_tag,
                class_="verdict-alert verdict-agree",
            )
        return ui.div(
            ui.tags.strong("Models disagree: "),
            "ResNet says ", pill(r_pred),
            ui.tags.span(f" ({r_conf:.0%})", style="margin: 0 0.5rem;"),
            "  ·  ViT says ", pill(v_pred),
            ui.tags.span(f" ({v_conf:.0%})", style="margin-left: 0.5rem;"),
            mods_tag,
            class_="verdict-alert verdict-disagree",
        )

    @render.plot
    def input_preview():
        result = inference()
        fig, ax = plt.subplots(figsize=(4, 4))
        if result is None:
            ax.text(0.5, 0.5, "No image selected",
                    ha="center", va="center", color="#aaa", fontsize=12)
        else:
            ax.imshow(result["rgb"])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("What the model sees", fontsize=11, color="#333")
        fig.tight_layout(pad=0.2)
        return fig

    def _prob_plot(model_key: str):
        result = inference()
        fig, ax = plt.subplots(figsize=(5.0, 2.4))
        if result is None:
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    color="#bbbbbb", fontsize=22)
            ax.axis("off")
            return fig

        probs = result["results"][model_key]["probs"]
        pred_idx = int(np.argmax(probs))
        colours = []
        for i in range(len(CLASSES)):
            base = CLASS_COLOURS[CLASSES[i]]
            colours.append(base if i == pred_idx else _fade(base, 0.20))

        bars = ax.barh(CLASSES, probs, color=colours, edgecolor="none", height=0.55)

        for bar, p, i in zip(bars, probs, range(len(CLASSES))):
            label_w = bar.get_width()
            ax.text(
                min(label_w + 0.02, 0.94),
                bar.get_y() + bar.get_height() / 2,
                f"{p:.0%}",
                va="center",
                fontsize=10,
                fontweight="bold" if i == pred_idx else "normal",
                color="#1f1f1f" if i == pred_idx else "#888888",
            )

        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1.0"], fontsize=9)
        ax.set_xlabel("Softmax probability", fontsize=9, color="#666")
        ax.tick_params(axis="y", length=0, labelsize=11)
        ax.tick_params(axis="x", length=2)
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"Class probabilities — Predicted: {CLASSES[pred_idx]}",
            fontsize=11, loc="left",
            color=CLASS_COLOURS[CLASSES[pred_idx]],
        )
        fig.subplots_adjust(left=0.22, right=0.96, top=0.78, bottom=0.28)
        return fig

    def _cam_plot(model_key: str):
        result = inference()
        fig, ax = plt.subplots(figsize=(4, 4))
        if result is None:
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    color="#bbbbbb", fontsize=22)
            ax.axis("off")
            return fig
        ax.imshow(result["rgb"])
        ax.imshow(result["results"][model_key]["cam"],
                  cmap="turbo", alpha=0.50, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Grad-CAM heatmap  (red = high attention)",
                     fontsize=11, color="#333")
        fig.tight_layout(pad=0.2)
        return fig

    @render.plot
    def resnet_probs():
        return _prob_plot("resnet50")

    @render.plot
    def resnet_cam():
        return _cam_plot("resnet50")

    @render.plot
    def vit_probs():
        return _prob_plot("vit_small_patch16_224")

    @render.plot
    def vit_cam():
        return _cam_plot("vit_small_patch16_224")


# Static asset mounts: serve assets/ at /assets and the project's figures/ at /figures
app = App(
    app_ui,
    server,
    static_assets={
        "/assets":  ASSETS_DIR,
        "/figures": FIGURES_DIR,
    },
)
