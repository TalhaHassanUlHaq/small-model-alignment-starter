# Task 3 — Supervised fine-tuning

## Goal

Fine-tune `Qwen/Qwen2.5-0.5B` on 30,000 instruction-response examples and
select a checkpoint using validation loss.

## Concepts

Study causal language modeling, teacher forcing, response-only loss masking,
padding, attention masks, gradient accumulation, effective batch size, fp16
autocast, gradient scaling, gradient clipping, validation, and checkpointing.

## Functions and pseudocode

### `format_prompt(instruction)`

1. Apply one documented prompt template.
2. Use it for both training and validation records.

### `build_lm_batch(tokenizer, prompts, responses, max_length)`

1. Tokenize each prompt and response separately.
2. Append EOS to each response.
3. Concatenate prompt and response tokens.
4. Truncate according to one documented policy.
5. Pad rows to a common length.
6. Set attention mask to one for real tokens and zero for padding.
7. Set prompt and padding labels to `-100`.
8. Copy response token IDs into the remaining label positions.

### `response_logprobs(model, batch)`

1. Run the causal model to obtain logits.
2. Shift logits and labels by one position.
3. Apply log-softmax over the vocabulary.
4. Gather the log probability of each target response token.
5. Ignore every label equal to `-100`.
6. Reduce consistently and return one score per example.

### `sft_loss(model, batch)`

1. Call `response_logprobs`.
2. Negate the scores.
3. Average using your chosen reduction.
4. Document the reduction in the report.

### `train_sft_step(...)`

1. Accumulate gradients across the configured number of mini-batches.
2. Use fp16 autocast on CUDA.
3. Scale the loss when using a gradient scaler.
4. Unscale before clipping.
5. Clip gradient norm.
6. Update the optimizer and clear gradients.

### `validate_sft(...)`

1. Switch the model to evaluation mode.
2. Disable gradients.
3. Compute the same response-only loss over validation batches.
4. Average and log the validation loss.
5. Restore training mode.

### `main()` in `scripts/train_sft.py`

1. Load and split the required 30k/1k examples.
2. Load tokenizer and model.
3. Create loaders, optimizer, scheduler, scaler, and metric storage.
4. Train for the selected epochs.
5. Validate at the chosen interval.
6. Save best and final checkpoints.
7. Save resolved configuration and metrics.

## Checklist

- Prompt tokens do not contribute to loss.
- Padding does not contribute to loss.
- Effective batch size equals per-device batch multiplied by accumulation.
- Trainable weights remain fp32 when using scaling.
- Best and final checkpoints are distinct artifacts when appropriate.
