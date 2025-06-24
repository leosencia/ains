# AI-No-Swiping: A Lightweight Adversarial Perturbation Tool to Protect Digital Artworks from AI Misuse
Authors: John Lawrence F. Quiñones, Dr. Maria Art Antonette D. Clariño, and Dr. Val Randolf M. Madrid

This paper presents AI-No-Swiping (AINS), a
lightweight adversarial perturbation tool that protects digital
artwork from unauthorized AI model training while maintaining
low computational requirements. As generative AI models in-
creasingly train on web-scraped artwork without artists’ consent,
existing protection tools remain inaccessible to many artists due
to high hardware demands (typically requiring specialized GPUs
with substantial VRAM). AINS addresses this limitation through
innovative memory optimization techniques—including tiled im-
age processing and half-precision model loading—enabling ef-
fective image protection on systems with only 4 GiB of VRAM.
Our implementation applies Projected Gradient Descent-based
adversarial perturbations based on Anti-DreamBooth’s Alter-
nating Surrogate and Perturbation Learning (ASPL) variation,
which disrupts diffusion models’ ability to learn artistic styles.
Experimental results using various datasets (clean, 100% per-
turbed, and mixed datasets with 10%, 25%, and 50% perturbed
images) demonstrate that AINS operates efficiently within target
memory constraints while effectively corrupting Stable Diffusion
models. Notably, our analysis of noise variance in homogeneous
areas reveals that even when only 10% of training images
contain AINS perturbations, significant disruption occurs in the
model’s learning capabilities. AINS offers artists with limited
computational resources a practical tool for protecting their
digital intellectual property in an increasingly AI-driven creative
landscape.

Keywords: adversarial perturbation, AI protection, digital
art protection, generative AI
