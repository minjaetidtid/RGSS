"""UNet from OpenAI guided-diffusion (https://github.com/openai/guided-diffusion), MIT License (see LICENSE).

Changes: AttentionBlock.forward in unet.py respects use_checkpoint (marked "local change:");
fp16_util.py keeps only the two helpers unet.py imports. nn.py is unmodified.
"""
