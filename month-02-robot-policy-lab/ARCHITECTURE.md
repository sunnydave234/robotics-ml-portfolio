# ACT Architecture Notes
# Answers sourced from pinned commit 2d7a420 — verify if LeRobot is re-pinned.

## Q1 — What is n_action_steps and where is it set?

File: src/lerobot/policies/act/configuration_act.py
Field: ACTConfig.n_action_steps (Python dataclass field, not a YAML key)
Default: 100
Meaning: How many future action steps the policy predicts per forward pass.
         The robot executes all n_action_steps before querying the policy again
         (when temporal ensembling is OFF, which is the default).
Constraint (__post_init__): n_action_steps <= chunk_size (default chunk_size=100).
                             n_action_steps must be 1 if temporal ensembling is ON.

## Q2 — Does __getitem__ return action shape (action_dim,) or (n_action_steps, action_dim)?

Answer: (n_action_steps, action_dim)
File: src/lerobot/policies/act/modeling_act.py
Evidence: Evidence: modeling_act.py line 121:
  # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor

This is the #1 silent failure in custom adapters.
(adim,) does NOT crash — PyTorch silently broadcasts it.
The policy trains, loss decreases, but the model learns a near-identity mapping
that fails completely at eval. Shape must be (n_action_steps, action_dim).

## Q3 — At eval, does the policy execute all n_action_steps before querying again?

Answer: YES (when temporal ensembling is disabled, which is the default).
File: src/lerobot/policies/act/modeling_act.py — select_action() method
Evidence: modeling_act.py lines 115-123:
  # Action queue logic for n_action_steps > 1.
  # When the action_queue is depleted, populate it by calling predict_action_chunk.
  if len(self._action_queue) == 0:
      actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
      self._action_queue.extend(actions.transpose(0, 1))
  return self._action_queue.popleft()

The queue has maxlen=n_action_steps (line 98). The policy executes ALL n_action_steps
actions before the queue empties and triggers a new forward pass.

Temporal ensembling (enabled via ACTConfig.use_temporal_ensemble=True) overlaps
inference windows and averages predictions. Default is OFF — full chunk executes.

## Q4 — What exact keys must a sample dict contain for ACT to train?

Required keys in __getitem__ return dict:

    "observation.image":  (C, H, W)                    float32, values in [0.0, 1.0]
    "observation.state":  (state_dim,)                  float32, normalized
    "action":             (n_action_steps, action_dim)  float32, normalized
Source: src/lerobot/policies/act/modeling_act.py — forward() input unpacking
        Evidence: modeling_act.py line 145:
  abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
  valid_mask = ~batch["action_is_pad"].unsqueeze(-1)

configuration_act.py docstring lines 34-40:
  - At least one key starting with "observation.image" required as input
  - May optionally work without "observation.state"
  - "action" is required as output key

Note: "action_is_pad" is also required by the loss function. LeRobotDataset generates
this automatically. Your RobotForgeAdapter must also generate it:
  "action_is_pad": (n_action_steps,) bool tensor — True for padded steps at episode end

Week 2 checklist — before submitting RobotForgeAdapter for review:
  [ ] action shape is (n_action_steps, action_dim), NOT (action_dim,)
  [ ] observation.image values are in [0.0, 1.0] float32, NOT uint8
  [ ] observation.state is normalized (z-score via dataset_stats.json)
  [ ] action is normalized (z-score via dataset_stats.json)
  [ ] All three keys present — no extras required, no keys missing


## pusht-specific notes (lerobot/pusht dataset)

action_dim: 2  (x, y end-effector target position)
state_dim:  2  (same as action in pusht — no proprioceptive joints)
image:      (3, 96, 96) float32

For RobotForgeAdapter wiring (Week 2):
- images: HDF5 /observations/images (T, H, W, C) uint8
  → convert to (C, H, W) float32 / 255.0 before returning
- state: HDF5 /observations/state (T, 2) float32
  → apply z-score normalization via dataset_stats.json
- action slice: actions[t : t + n_action_steps] from HDF5 /actions (T, 2)
  → shape must be (n_action_steps, 2) — NOT (2,)
  → episode boundary: if t + n_action_steps > episode_end, repeat-pad the last action
- action_is_pad: (n_action_steps,) bool tensor
  → False for real steps, True for repeat-padded steps past episode end
  → required by loss function (lerobot_train.py line 145) — missing key = KeyError at training start

##hook points (pinned commit 2d7a420)

For Week 2 adapter wiring:
- Dataset created:  make_train_eval_datasets(cfg) — line 249
  RobotForgeAdapter must be interface-compatible with what this returns
- Forward pass:     policy.forward(batch) — line 136 (normal), line 121 (weighted/per-sample)
  Week 4 hard-example mining uses the reduction="none" path — already exists, just needs wiring
- Eval (loss):      is_eval_step — line 585, runs policy.forward on eval_dataloader, no env needed
- Eval (rollout):   is_env_eval_step — line 584, calls eval_policy_all(), returns pc_success + video
- Checkpoint save:  save_checkpoint() — line 643, resume path at line 380
- Device:           accelerator.device — line 236, auto-detects MPS via HuggingFace accelerate
  pin_memory=device.type=="cuda" — line 460, correctly False on MPS (unified memory, no-op)
  [ ] action_is_pad present: (n_action_steps,) bool, True for padded steps past episode end
