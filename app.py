"""ToolsTap AI backend for Hugging Face ZeroGPU Spaces.

This app intentionally uses the Gradio SDK. ZeroGPU does not support Docker/FastAPI-only
Spaces. The three functions bound to Gradio events are directly decorated with @spaces.GPU
so the ZeroGPU scheduler can allocate a GPU for inference.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HOME", "/tmp/toolstap_hf_cache")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/toolstap_hf_cache/modules")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/toolstap_hf_cache/transformers")

# IMPORTANT: import spaces before torch, as required by ZeroGPU.
import spaces  # provided by the Hugging Face ZeroGPU runtime; don't pin in requirements.txt
import torch
import gradio as gr
import cv2
import numpy as np
import re
import sys
import time
import uuid
import urllib.request
from pathlib import Path
from typing import Any
from PIL import Image, ImageColor, ImageChops, ImageOps, ImageFilter, UnidentifiedImageError

APP_NAME = "ToolsTap AI Processing API"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SOURCE_PIXELS = 20_000_000
MAX_OUTPUT_PIXELS = 49_000_000
MAX_OUTPUT_LONG_EDGE = 12_000
ARTIFACT_TTL_SECONDS = 40 * 60
OUTPUT_DIR = Path("/tmp/toolstap-ai-results")
MODEL_DIR = Path("/tmp/toolstap-ai-models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"

print(f"[{APP_NAME}] device={DEVICE}, half_precision={USE_HALF}", flush=True)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Fix compatibility with BasicSR versions that import a removed torchvision internal module.
try:
    import torchvision.transforms.functional_tensor  # type: ignore # noqa: F401
except Exception:
    try:
        import torchvision.transforms.functional as tv_functional
        sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)
    except Exception:
        pass


def _clean_old_files() -> None:
    cutoff = time.time() - ARTIFACT_TTL_SECONDS
    for folder in (OUTPUT_DIR,):
        try:
            for path in folder.iterdir():
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass


def _safe_stem(filename: str | None, default: str = "image") -> str:
    stem = Path(filename or default).stem or default
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")
    return (stem or default)[:90]


def _read_image(file_path: str | None) -> tuple[Image.Image, str]:
    if not file_path or not os.path.isfile(file_path):
        raise gr.Error("Please upload an image first.")
    if os.path.getsize(file_path) > MAX_UPLOAD_BYTES:
        raise gr.Error("Image is larger than 25 MB. Choose a smaller file.")
    try:
        with Image.open(file_path) as opened:
            fmt = (opened.format or Path(file_path).suffix.lstrip(".")).lower()
            if fmt not in {"png", "jpeg", "jpg", "webp", "bmp"}:
                raise gr.Error("Supported formats: PNG, JPG, WebP and BMP.")
            image = ImageOps.exif_transpose(opened).copy()
            image.load()
    except gr.Error:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise gr.Error("The uploaded image is invalid or could not be decoded.") from exc
    if image.width * image.height > MAX_SOURCE_PIXELS:
        raise gr.Error(f"Image is {image.width:,} × {image.height:,}. Server limit is {MAX_SOURCE_PIXELS // 1_000_000} megapixels per source image.")
    return image, fmt


def _output_dimensions(width: int, height: int, factor: float) -> tuple[int, int]:
    out_w = max(1, int(round(width * factor)))
    out_h = max(1, int(round(height * factor)))
    if out_w * out_h > MAX_OUTPUT_PIXELS:
        raise gr.Error(f"Requested result is {out_w:,} × {out_h:,}. Safe output limit is 49 megapixels. Choose a smaller scale.")
    if max(out_w, out_h) > MAX_OUTPUT_LONG_EDGE:
        raise gr.Error("The maximum output long edge is 12,000 pixels.")
    return out_w, out_h


def _save_result(image: Image.Image, filename: str, output_format: str) -> tuple[str, str, dict[str, Any]]:
    _clean_old_files()
    fmt = (output_format or "png").lower()
    if fmt in {"jpg", "jpeg"}:
        fmt = "jpg"
    elif fmt not in {"png", "webp"}:
        fmt = "png"
    stem = _safe_stem(filename)
    file_name = f"{stem}.{fmt}"
    token = uuid.uuid4().hex
    full_path = OUTPUT_DIR / f"{token}-{file_name}"
    preview_path = OUTPUT_DIR / f"{token}-preview.webp"

    final = image
    if fmt == "jpg":
        if final.mode in {"RGBA", "LA"}:
            rgba = final.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            final = background.convert("RGB")
        else:
            final = final.convert("RGB")
        final.save(full_path, format="JPEG", quality=94, optimize=True, progressive=True)
        mime_type = "image/jpeg"
    elif fmt == "webp":
        final = final.convert("RGBA") if final.mode in {"RGBA", "LA"} else final.convert("RGB")
        final.save(full_path, format="WEBP", quality=94, method=4)
        mime_type = "image/webp"
    else:
        final = final.convert("RGBA") if final.mode in {"RGBA", "LA"} else final.convert("RGB")
        final.save(full_path, format="PNG", optimize=False, compress_level=3)
        mime_type = "image/png"

    preview = final.copy()
    preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    preview.save(preview_path, format="WEBP", quality=82, method=4)
    meta = {
        "filename": file_name,
        "width": final.width,
        "height": final.height,
        "size_bytes": full_path.stat().st_size,
        "mime_type": mime_type,
        "note": "Preview is resized for quick display. Download contains the full-resolution output.",
    }
    return str(preview_path), str(full_path), meta


# Background removal model. BiRefNet is MIT-licensed; model details: https://huggingface.co/ZhengPeng7/BiRefNet
# ZeroGPU requires the model to be placed on cuda at module scope, after importing `spaces`.
BG_MODEL = None
BG_MODEL_ERROR = None
try:
    import transformers
    # BiRefNet community demo applies this compatibility setting for its custom remote model code.
    try:
        torch.jit.script = lambda fn: fn  # type: ignore[assignment]
    except Exception:
        pass
    from transformers import AutoModelForImageSegmentation
    BG_MODEL = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    BG_MODEL.to(DEVICE)
    BG_MODEL.eval()
    if USE_HALF:
        BG_MODEL.half()
    print("[ToolsTap] BiRefNet background model loaded.", flush=True)
except Exception as exc:
    BG_MODEL_ERROR = f"{type(exc).__name__}: {exc}"
    BG_MODEL = None
    print(f"[ToolsTap] Background model failed to load: {BG_MODEL_ERROR}", flush=True)


def _download_weight(model_key: str) -> Path:
    urls = {
        "general": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth",
        "anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus_anime_6B.pth",
    }
    names = {
        "general": "RealESRGAN_x4plus.pth",
        "anime": "RealESRGAN_x4plus_anime_6B.pth",
    }
    target = MODEL_DIR / names[model_key]
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"[ToolsTap] Downloading {names[model_key]} model weights…", flush=True)
    try:
        with urllib.request.urlopen(urls[model_key], timeout=180) as response, open(tmp, "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if tmp.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model weights were incomplete.")
        tmp.replace(target)
        return target
    finally:
        tmp.unlink(missing_ok=True)


UPSAMPLERS: dict[str, Any] = {}
UPSCALER_ERRORS: dict[str, str] = {}
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    def _build_upsampler(key: str):
        weight_path = _download_weight(key)
        if key == "anime":
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        else:
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        return RealESRGANer(
            scale=4,
            model_path=str(weight_path),
            model=net,
            tile=256,
            tile_pad=16,
            pre_pad=0,
            half=USE_HALF,
            device=torch.device(DEVICE),
        )

    # Load at module scope so ZeroGPU can pack model weights before requests arrive.
    for _key in ("general", "anime"):
        try:
            UPSAMPLERS[_key] = _build_upsampler(_key)
            print(f"[ToolsTap] Real-ESRGAN {_key} model loaded.", flush=True)
        except Exception as exc:
            UPSCALER_ERRORS[_key] = f"{type(exc).__name__}: {exc}"
            print(f"[ToolsTap] Real-ESRGAN {_key} model failed: {UPSCALER_ERRORS[_key]}", flush=True)
except Exception as exc:
    for _key in ("general", "anime"):
        UPSCALER_ERRORS[_key] = f"{type(exc).__name__}: {exc}"
    print(f"[ToolsTap] Real-ESRGAN library import failed: {type(exc).__name__}: {exc}", flush=True)


def _set_background(image_rgba: Image.Image, background: str, output_format: str) -> Image.Image:
    choice = (background or "transparent").strip().lower()
    if choice in {"transparent", "none", "no background"}:
        if output_format.lower() in {"jpg", "jpeg"}:
            white = Image.new("RGBA", image_rgba.size, (255, 255, 255, 255))
            white.alpha_composite(image_rgba)
            return white.convert("RGB")
        return image_rgba
    try:
        color = ImageColor.getrgb(choice)
        if len(color) == 3:
            color = (*color, 255)
    except ValueError:
        color = (255, 255, 255, 255)
    bg = Image.new("RGBA", image_rgba.size, color)
    bg.alpha_composite(image_rgba)
    return bg.convert("RGB") if output_format.lower() in {"jpg", "jpeg"} else bg


@spaces.GPU(duration=120)
def remove_background(file_path: str, model: str = "fast", background: str = "transparent", output_format: str = "png", filename: str = "image.png"):
    if BG_MODEL is None:
        detail = BG_MODEL_ERROR or "Model not available"
        raise gr.Error(f"Background AI model could not start. Space logs: {detail[:220]}")
    image, _ = _read_image(file_path)
    source = image.convert("RGB")
    resolution = 1536 if str(model).lower() in {"quality", "detail", "high-detail"} else 1024
    from torchvision import transforms
    prep = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = prep(source).unsqueeze(0).to(DEVICE)
    if USE_HALF:
        tensor = tensor.half()
    try:
        with torch.inference_mode():
            predictions = BG_MODEL(tensor)[-1].sigmoid().float().cpu()
        mask = transforms.ToPILImage()(predictions[0].squeeze()).resize(source.size, Image.Resampling.LANCZOS)
        # Mild antialiasing prevents jagged alpha edges without making the subject blurry.
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.35))
        rgba = image.convert("RGBA")
        if "A" in image.getbands():
            mask = ImageChops.multiply(mask, image.getchannel("A"))
        rgba.putalpha(mask)
        final = _set_background(rgba, background, output_format)
        stem = _safe_stem(filename or Path(file_path).name)
        fmt = (output_format or "png").lower()
        if fmt not in {"png", "webp", "jpg", "jpeg"}:
            fmt = "png"
        preview_path, full_path, meta = _save_result(final, f"{stem}-background-removed.png", fmt)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return preview_path, full_path, meta
    finally:
        del tensor
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def _prepare_cv(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    rgba = image.convert("RGBA")
    rgba_arr = np.array(rgba, dtype=np.uint8)
    alpha = rgba_arr[:, :, 3].copy() if not np.all(rgba_arr[:, :, 3] == 255) else None
    return cv2.cvtColor(rgba_arr[:, :, :3], cv2.COLOR_RGB2BGR), alpha


def _finish_cv(bgr: np.ndarray, alpha: np.ndarray | None, out_w: int, out_h: int) -> Image.Image:
    if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if alpha is not None:
        if alpha.shape[:2] != (out_h, out_w):
            alpha = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    return Image.fromarray(rgb, "RGB")


def _format_choice(fmt: str) -> str:
    fmt = (fmt or "png").lower()
    if fmt in {"jpg", "jpeg"}:
        return "jpg"
    return fmt if fmt in {"png", "webp"} else "png"


def _upscale_core(file_path: str, scale: str | float, model: str, output_format: str, filename: str, line_art: bool = False, output_long_edge: str | int = "0", edge_mode: str = "balanced", sharpness: str | float = 0.45, bw_cleanup: bool = False, threshold: str | int = 170, use_ai: bool = True):
    key = "anime" if line_art or model == "anime" else "general"
    upsampler = UPSAMPLERS.get(key)
    if upsampler is None and use_ai:
        error = UPSCALER_ERRORS.get(key, "model is unavailable")
        raise gr.Error(f"AI upscaler model could not start ({error[:180]}). Open Space → Logs to see the build/runtime error.")

    image, _ = _read_image(file_path)
    source_w, source_h = image.size
    bgr, alpha = _prepare_cv(image)
    target = int(float(output_long_edge or 0)) if line_art else 0
    if line_art and target > 0:
        factor = max(1.0, target / max(source_w, source_h))
    else:
        factor = max(2.0, min(4.0, float(scale or 2)))
    out_w, out_h = _output_dimensions(source_w, source_h, factor)

    if use_ai and factor > 1.01:
        with torch.inference_mode():
            output, _ = upsampler.enhance(bgr, outscale=factor)
    else:
        output = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    if line_art:
        mode = (edge_mode or "balanced").lower()
        amount = max(0.0, min(1.5, float(sharpness or 0.45)))
        if mode == "smooth":
            output = cv2.bilateralFilter(output, d=5, sigmaColor=28, sigmaSpace=28)
        elif mode == "rounded":
            blurred = cv2.GaussianBlur(output, (3, 3), 0.7)
            output = cv2.addWeighted(output, 0.82, blurred, 0.18, 0)
        elif mode == "sharp":
            blurred = cv2.GaussianBlur(output, (0, 0), 1.0)
            output = cv2.addWeighted(output, 1.0 + 0.75 * amount, blurred, -0.75 * amount, 0)
        elif mode == "balanced" and amount > 0.05:
            blurred = cv2.GaussianBlur(output, (0, 0), 0.7)
            output = cv2.addWeighted(output, 1.0 + 0.28 * amount, blurred, -0.28 * amount, 0)
        if str(bw_cleanup).lower() in {"true", "1", "yes"}:
            cutoff = max(0, min(255, int(float(threshold or 170))))
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            _, output = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    final = _finish_cv(output, alpha, out_w, out_h)
    fmt = _format_choice(output_format)
    stem = _safe_stem(filename or Path(file_path).name)
    preview_path, full_path, meta = _save_result(final, f"{stem}-upscaled.png", fmt)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return preview_path, full_path, meta


@spaces.GPU(duration=300)
def upscale_image(file_path: str, scale: str = "2", model: str = "general", output_format: str = "png", filename: str = "image.png"):
    return _upscale_core(file_path, scale, model, output_format, filename, line_art=False)


@spaces.GPU(duration=300)
def line_art_upscale(file_path: str, scale: str = "2", output_long_edge: str = "0", edge_mode: str = "balanced", sharpness: str = "0.45", bw_cleanup: bool = False, threshold: str = "170", use_ai: bool = True, filename: str = "line-art.png"):
    return _upscale_core(
        file_path, scale, "anime", "png", filename, line_art=True,
        output_long_edge=output_long_edge, edge_mode=edge_mode,
        sharpness=sharpness, bw_cleanup=bw_cleanup, threshold=threshold,
        use_ai=use_ai,
    )


def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "sdk": "gradio-zero-gpu",
        "device": DEVICE,
        "background_model_loaded": BG_MODEL is not None,
        "upscaler_general_loaded": "general" in UPSAMPLERS,
        "upscaler_anime_loaded": "anime" in UPSAMPLERS,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "max_output_megapixels": MAX_OUTPUT_PIXELS / 1_000_000,
    }


with gr.Blocks(title="ToolsTap AI Backend", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# ToolsTap AI Processing Backend\nThis public API powers the ToolsTap website's background remover and image upscalers. Large files are processed on the server; output previews are downsized.")
    with gr.Tab("Background Remover"):
        bg_file = gr.File(label="Image file", type="filepath", file_types=["image"])
        bg_model = gr.Radio(choices=["fast", "quality"], value="fast", label="Model mode")
        bg_color = gr.Textbox(value="transparent", label="Background: transparent, white, black or #RRGGBB")
        bg_format = gr.Dropdown(choices=["png", "webp", "jpg"], value="png", label="Output format")
        bg_filename = gr.Textbox(value="image.png", label="Output filename")
        bg_button = gr.Button("Remove background", variant="primary")
        bg_preview = gr.Image(type="filepath", label="Result preview")
        bg_download = gr.File(label="Download full-size result")
        bg_meta = gr.JSON(label="Result information")
        bg_button.click(remove_background, inputs=[bg_file, bg_model, bg_color, bg_format, bg_filename], outputs=[bg_preview, bg_download, bg_meta], api_name="remove_background", concurrency_limit=1)

    with gr.Tab("AI Image Upscaler"):
        up_file = gr.File(label="Image file", type="filepath", file_types=["image"])
        up_scale = gr.Dropdown(choices=["2", "4"], value="2", label="Scale")
        up_model = gr.Dropdown(choices=["general", "anime"], value="general", label="Model")
        up_format = gr.Dropdown(choices=["png", "webp", "jpg"], value="png", label="Output format")
        up_filename = gr.Textbox(value="image.png", label="Output filename")
        up_button = gr.Button("Upscale image", variant="primary")
        up_preview = gr.Image(type="filepath", label="Result preview")
        up_download = gr.File(label="Download full-size result")
        up_meta = gr.JSON(label="Result information")
        up_button.click(upscale_image, inputs=[up_file, up_scale, up_model, up_format, up_filename], outputs=[up_preview, up_download, up_meta], api_name="upscale_image", concurrency_limit=1)

    with gr.Tab("Line Art Upscaler"):
        line_file = gr.File(label="Image file", type="filepath", file_types=["image"])
        line_scale = gr.Dropdown(choices=["2", "3", "4"], value="2", label="Scale if custom edge is off")
        line_edge = gr.Dropdown(choices=["0", "2000", "3000", "5000", "7000"], value="0", label="Output long edge (pixels)")
        line_mode = gr.Dropdown(choices=["balanced", "smooth", "sharp", "rounded"], value="balanced", label="Edge treatment")
        line_sharpness = gr.Slider(minimum=0.0, maximum=1.5, value=0.45, step=0.05, label="Sharpness")
        line_bw = gr.Checkbox(value=False, label="Clean to black-and-white")
        line_threshold = gr.Slider(minimum=0, maximum=255, value=170, step=1, label="Black/white threshold")
        line_ai = gr.Checkbox(value=True, label="Use AI enhancement")
        line_filename = gr.Textbox(value="line-art.png", label="Output filename")
        line_button = gr.Button("Upscale line art", variant="primary")
        line_preview = gr.Image(type="filepath", label="Result preview")
        line_download = gr.File(label="Download full-size result")
        line_meta = gr.JSON(label="Result information")
        line_button.click(line_art_upscale, inputs=[line_file, line_scale, line_edge, line_mode, line_sharpness, line_bw, line_threshold, line_ai, line_filename], outputs=[line_preview, line_download, line_meta], api_name="line_art_upscale", concurrency_limit=1)

    gr.Markdown("### Health status\nUse the API page's `health_check` endpoint to confirm model loading. A working Space does not mean ZeroGPU quota is unlimited; requests can queue or fail when shared quota is exhausted.")
    health_button = gr.Button("Check backend health")
    health_json = gr.JSON(label="Health")
    health_button.click(health_check, inputs=[], outputs=[health_json], api_name="health_check", concurrency_limit=1)

demo.queue(default_concurrency_limit=1, max_size=10)

if __name__ == "__main__":
    demo.launch(analytics_enabled=False, max_file_size="25mb", delete_cache=(1800, 2400))
