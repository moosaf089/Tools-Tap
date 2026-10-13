---
title: ToolsTap AI Backend
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.50.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# ToolsTap AI Backend

Public Gradio API for the ToolsTap website. This Space is designed for Hugging Face ZeroGPU; use the ZeroGPU hardware setting and keep the Gradio SDK in this front matter.

## API functions

- `/remove_background`: AI foreground mask and transparent/color background export.
- `/upscale_image`: Real-ESRGAN super-resolution (general or anime/illustration model).
- `/line_art_upscale`: Real-ESRGAN anime model plus configurable edge treatment.
- `/health_check`: Reports whether the models loaded successfully.

Open the Space's **Use via API** page to inspect input/output schemas. The public ToolsTap website calls these endpoints through `@gradio/client`; do not paste a personal Hugging Face token into the public website.

## Operational limitations

- First startup downloads model weights and can take several minutes.
- Each inference function requests a ZeroGPU allocation. Requests may queue or fail when the free shared quota is exhausted.
- Maximum source image is 25 MB / 20 megapixels. Maximum output is 49 megapixels and 12,000 pixels on the long edge.
- Results and previews are temporary Gradio-cached files; the Space removes expired output files during subsequent jobs and Gradio clears its cache on its own schedule.
- Public API access is not unlimited. Add abuse protection and monitor usage before promoting this as a high-traffic service.
