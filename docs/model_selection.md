# Model Selection

## Recommendation

Use Qwen3-VL small model as the Week 1 primary candidate. If the checkpoint, vLLM support, or local VRAM blocks progress, fall back to Qwen2.5-VL-3B-Instruct for the engineering pipeline.

DeepSeek-VL2 Tiny / Small should be treated as a comparison candidate after the first pipeline is stable.

## Comparison

| Dimension | Qwen-VL / Qwen3-VL | DeepSeek-VL2 Tiny / Small |
|---|---|---|
| VRAM | Small variants are suitable for local-first attempts | Tiny/Small active parameters are smaller, but MoE loading can be more complex |
| Deployment | Better first-week choice because vLLM support path is clearer | Validate vLLM support before using as serving baseline |
| Chinese | Strong for Chinese OTA prompts | Strong, but serving ecosystem needs validation |
| OCR | Good fit for signs, menus, shop names | Paper emphasizes OCR/document/chart capability |
| Product/POI recognition | Good fit for structured extraction | Good comparison model for error analysis |
| Multi-image | Suitable target for multi-image tests | Needs project-specific verification |
| First-week fit | Primary | Secondary |

## First Run Policy

1. Try `Qwen/Qwen3-VL-2B-Instruct`.
2. If blocked, use `Qwen/Qwen2.5-VL-3B-Instruct`.
3. Record exact model, vLLM version, command, GPU memory, prompt, and output.
4. Do not spend Week 1 tuning quality before the API and experiment loop work.

