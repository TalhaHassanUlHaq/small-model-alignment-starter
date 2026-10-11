# Task 4 — From-scratch DPO

## Goal

Run the same DPO implementation twice: once on 15,000 HH-RLHF-style pairs and
once on 15,000 UltraFeedback-style pairs. Start both runs from the same best
SFT checkpoint.

## Concepts

Study preference pairs, policy/reference models, implicit rewards, response
sequence scores, beta, logistic preference loss, reward accuracy, reward
margin, frozen parameters, and dataset-specific alignment behavior.

## Functions and pseudocode

### `build_preference_batch(...)`

1. Build one response-masked language-model batch for chosen responses.
2. Build another for rejected responses.
3. Keep row order and prompts aligned between both batches.

### `compute_dpo_loss_from_logps(...)`

1. Compute the policy chosen-minus-rejected gap.
2. Compute the reference chosen-minus-rejected gap.
3. Subtract the reference gap from the policy gap.
4. Multiply by beta.
5. Apply negative log-sigmoid.
6. Average across examples.

### `compute_dpo_loss(...)`

1. Score chosen and rejected responses with the policy.
2. Score both responses with the reference under no-gradient evaluation.
3. Pass all four scores to the tensor-level DPO function.
4. Return one scalar loss.

### `dpo_batch_step(...)`

1. Compute policy and reference response scores.
2. Compute DPO loss and backpropagate only through policy scores.
3. Accumulate gradients as configured.
4. Clip policy gradients and update the optimizer.
5. Compute implicit chosen/rejected rewards.
6. Return loss, strict reward accuracy, and mean reward margin.

### `main()` in `scripts/train_dpo.py`

1. Select one preference dataset track.
2. Load the prescribed train/validation counts.
3. Load policy and frozen reference from the SFT-best checkpoint.
4. Run training and validation.
5. Select the best checkpoint by validation reward accuracy.
6. Save final checkpoint, metrics, configuration, and dataset identity.
7. Repeat independently for the second dataset.

## Checklist

- Reference weights never receive gradients or optimizer updates.
- Prompt and padding tokens are excluded from sequence scores.
- Both runs begin from the same SFT checkpoint.
- Strict ties count as reward accuracy zero, as specified by the manual.
- HH-RLHF and UltraFeedback results remain separate.
