import os
import json
from abc import ABC
from typing import Dict, Optional, Union

import ray
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import ValueLoss, get_llm_for_sequence_regression
from openrlhf.models.utils import masked_mean
from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import (
    offload_deepspeed_states,
    reload_deepspeed_states,
)
from openrlhf.utils.loss_utils import get_loss_batch_info

from ..ppo_utils import NaiveReplayBuffer
from .launcher import BaseModelActor

_METRICS_PATH = os.environ.get("PPO_METRICS_FILE", "runs/ppo_metrics.jsonl")


def _activation_offload_ctx():
    import torch.autograd.graph as _g

    def _pack(t):
        return t.detach().to("cpu")

    def _unpack(t):
        return t.contiguous().to(torch.cuda.current_device())

    return _g.saved_tensors_hooks(_pack, _unpack)


def _patch_ds_config_cpu_checkpointing():
    import openrlhf.utils.deepspeed.deepspeed_utils as _du
    if getattr(_du, "_cpu_ckpt_patched", False):
        return
    _orig = _du.get_train_ds_config

    def _patched(*args, **kwargs):
        cfg = _orig(*args, **kwargs)
        ac = cfg.setdefault("activation_checkpointing", {})
        ac["cpu_checkpointing"] = True
        ac["partition_activations"] = False
        return cfg

    _du.get_train_ds_config = _patched
    _du._cpu_ckpt_patched = True


def _prewarm_adam_states(engine_optim):
    base = getattr(engine_optim, "optimizer", engine_optim)
    for group in getattr(base, "param_groups", []):
        for p in group.get("params", []):
            if p.requires_grad:
                st = base.state[p]
                if "exp_avg" not in st or st["exp_avg"] is None:
                    st["exp_avg"] = torch.zeros_like(p.data)
                if "exp_avg_sq" not in st or st["exp_avg_sq"] is None:
                    st["exp_avg_sq"] = torch.zeros_like(p.data)


def _replace_checkpoint_func(model):
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint as _ckpt_fn
    except ImportError:
        try:
            from deepspeed.checkpointing import checkpoint as _ckpt_fn
        except ImportError:
            print("[WARN] deepspeed checkpointing unavailable; skipping activation offload (route 2)")
            return
    for _m in model.modules():
        if hasattr(_m, "_gradient_checkpointing_func"):
            _m._gradient_checkpointing_func = _ckpt_fn


def _append_metrics(entry: dict) -> None:
    try:
        with open(_METRICS_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write metrics to {_METRICS_PATH}: {e}")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_step_accuracy(experience) -> float:
    """Per-step accuracy from experience.info, returns scalar float."""
    info = getattr(experience, "info", {}) or {}
    for key in ("accuracy", "accuracies", "acc", "correct", "is_correct"):
        val = info.get(key)
        if val is not None:
            try:
                if isinstance(val, torch.Tensor):
                    val = val.detach().float().mean().item()
                return float(val)
            except (TypeError, ValueError):
                continue
    # Fallback: GSM8K 0/1 reward equals accuracy
    rewards = getattr(experience, "rewards", None)
    if rewards is not None:
        try:
            if isinstance(rewards, torch.Tensor):
                return rewards.detach().float().mean().item()
        except (TypeError, ValueError):
            pass
    return 0.0


def _compute_step_kl(experience) -> float:
    """Per-step |KL| from action_log_probs - base_action_log_probs, returns scalar float."""
    alp = getattr(experience, "action_log_probs", None)
    blp = getattr(experience, "base_action_log_probs", None)
    if alp is None or blp is None:
        return 0.0
    mask = experience.action_mask.bool()
    diff = (alp.float() - blp.float())[mask]
    return diff.abs().mean().item()


def _compute_step_entropy(experience) -> float:
    """Per-step entropy proxy: -mean(action_log_probs) over action tokens.

    True entropy (from full logits distribution) is only available in the actor's
    forward pass.  The critic receives its own Ray-serialized copy of Experience
    before the actor runs, so experience.info["entropy"] is always absent here.
    We use negated mean log-prob as a proxy — for well-calibrated models it is
    directionally consistent with true token entropy (lower = more confident).
    """
    alp = getattr(experience, "action_log_probs", None)
    if alp is None:
        return 0.0
    mask = experience.action_mask.bool()
    lp = alp.float()[mask]
    if lp.numel() == 0:
        return 0.0
    # Proxy: negated mean log-prob.  Typical range [0.1, 5.0] for LM tokens.
    return -lp.mean().item()


def _entropy_collapse(entropy: float, entropy_ema: float) -> float:
    if entropy <= 0 or entropy_ema <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, (entropy_ema - entropy) / entropy_ema))


def _prune_hf_ckpts(root: str, keep: int) -> None:
    import re as _re
    import shutil as _shutil

    if keep <= 0 or not os.path.isdir(root):
        return
    items = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        m = _re.findall(r"(\d+)", name)
        if m:
            items.append((int(m[-1]), p))
    items.sort()
    for _, p in items[:max(0, len(items) - keep)]:
        try:
            _shutil.rmtree(p)
        except OSError:
            pass


class CriticPPOTrainerStar(ABC):
    def __init__(
        self,
        strategy,
        critic: torch.nn.Module,
        critic_optim: Optimizer,
        critic_scheduler,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        value_clip: float = 0.2,
        dataloader_pin_memory: bool = True,
        few_shot_examples: list = None,  # list of tensors, fixed 2 examples
        total_train_steps: int = 2048,
        answer_map_path: str = "",  # path to JSONL with {question, answer} for privileged info
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_scheduler = critic_scheduler
        self.micro_train_batch_size = micro_train_batch_size
        self.buffer_limit = buffer_limit
        self.buffer_cpu_offload = buffer_cpu_offload
        self.value_clip = value_clip
        self.dataloader_pin_memory = dataloader_pin_memory
        self.few_shot_examples = few_shot_examples
        self.total_train_steps = total_train_steps

        # ── Load answer map for privileged (q, a) conditioning (OPSD-style) ──
        self.answer_map: dict[str, str] = {}
        self._tokenizer_for_answer = None  # lazily set
        if answer_map_path and os.path.exists(answer_map_path):
            try:
                with open(answer_map_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        q = item.get("question", "").strip()
                        a = item.get("answer", "").strip()
                        if q and a:
                            self.answer_map[q] = a
                strategy.print(f"[AdaptivePPO*] Loaded {len(self.answer_map)} answer mappings from {answer_map_path}")
            except Exception as e:
                strategy.print(f"[AdaptivePPO*] Failed to load answer map: {e}")
        self.max_epochs = self.args.train.max_epochs

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            self.args.ds.packing_samples,
            self.args.train.dynamic_batch_enable,
        )

        self.critic_loss_fn = ValueLoss(value_clip)

        # Mixtral 8x7b
        self.aux_loss = self.args.actor.aux_loss_coef > 1e-8

        # ── Per-step Adaptive Critic state ──
        self._consecutive_skips: int = 0
        self._prev_kl: float = 0.0
        self._total_steps: int = 0
        self._total_updates: int = 0
        self._entropy_ema: float = 0.0
        self._acc_ema: float = 0.0
        self._acc_history: list[float] = []

        self._inject_total: int = 0
        self._inject_attempted: int = 0
        self._inject_hit: int = 0

        self._privilege_prob_enabled = True

    def _step_adaptive_decision(self, step_kl: float, step_accuracy: float, step_entropy: float) -> dict:
        """Per-step three-track AND gate + safety nets."""
        args = self.args

        # ── Track-1: accuracy growth rate (smoothed before diff) ──
        # EMA-smooth raw per-step accuracy first, then put smoothed value
        # into the sliding window.  This prevents single-step {0.0,0.5,1.0}
        # jumps from dominating the growth computation (≈ 3-step effective window).
        self._acc_ema = 0.5 * self._acc_ema + 0.5 * step_accuracy
        self._acc_history.append(self._acc_ema)
        window = getattr(args, "adaptive_acc_growth_window", 5)
        if len(self._acc_history) > window:
            self._acc_history.pop(0)
        if len(self._acc_history) >= 2:
            acc_growth = (self._acc_history[-1] - self._acc_history[0]) / max(len(self._acc_history) - 1, 1)
        else:
            acc_growth = -1.0

        # ── Track-3: entropy EMA & collapse ──
        if step_entropy > 0:
            self._entropy_ema = 0.9 * self._entropy_ema + 0.1 * step_entropy
        entropy_ema = self._entropy_ema
        entropy_collapse = _entropy_collapse(step_entropy, entropy_ema)

        # ── Thresholds (set via CLI, no hardcoded fallback) ──
        acc_growth_th = getattr(args, "adaptive_acc_growth_threshold")
        kl_th = getattr(args, "adaptive_kl_threshold")
        ent_th = getattr(args, "adaptive_entropy_threshold")
        kl_ceiling = getattr(args, "adaptive_kl_ceiling")
        kl_delta = getattr(args, "adaptive_kl_delta")
        max_skip = getattr(args, "adaptive_max_skip")
        ent_floor = getattr(args, "adaptive_ent_floor")


        # ── Component 1: Three-Track AND Gate ──
        three_track = (
            acc_growth < acc_growth_th
            and step_kl > kl_th
            and step_entropy < ent_th
        )
        critic_updated = three_track
        reason = "three_track" if critic_updated else ""
        reason_id = 1 if critic_updated else 0

        # ── Component 2a: KL ceiling ──
        if not critic_updated and step_kl > kl_ceiling:
            critic_updated = True
            reason = "kl_ceiling"
            reason_id = 2

        # ── Component 3a: Max-skip guard (step-level) ──
        if not critic_updated and max_skip > 0 and self._consecutive_skips >= max_skip:
            critic_updated = True
            reason = "max_skip"
            reason_id = 3

        # ── Component 2b: KL delta surge ──
        if not critic_updated:
            if kl_delta > 0 and step_kl > 0.015:
                prev_kl = self._prev_kl
                kl_jump = abs(step_kl - prev_kl)
                relative_jump = kl_jump / max(prev_kl, 1e-5)
                if relative_jump > kl_delta:
                    critic_updated = True
                    reason = "kl_delta"
                    reason_id = 4

        # ── Component 3b: Entropy floor ──
        if not critic_updated and ent_floor > 0:
            if step_entropy < ent_floor:
                critic_updated = True
                reason = "ent_low"
                reason_id = 5

        # ── Update persistent state ──
        self._prev_kl = step_kl
        self._total_steps += 1
        if critic_updated:
            self._consecutive_skips = 0
            self._total_updates += 1
        else:
            self._consecutive_skips += 1

        return {
            "critic_updated": critic_updated,
            "reason": reason,
            "reason_id": reason_id,
            "kl": step_kl,
            "accuracy": step_accuracy,
            "acc_growth": acc_growth,
            "entropy": step_entropy,
            "entropy_ema": entropy_ema,
            "entropy_collapse": entropy_collapse,
            "three_track_trigger": three_track,
            "three_track_acc_ok": acc_growth < acc_growth_th,
            "three_track_kl_ok": step_kl > kl_th,
            "three_track_ent_ok": step_entropy < ent_th,
            "total_updates": self._total_updates,
            "total_steps": self._total_steps,
            "consecutive_skips": self._consecutive_skips,
        }

    def _progress(self) -> float:
        return self._total_steps / max(self.total_train_steps, 1)

    def _should_inject(self) -> bool:
        return False

    def _inject_prob(self, mode: str) -> float:
        progress = self._progress()
        # 0.5 at progress 0 -> 0 at progress 0.25, then constant 0.
        return 0.5 * max(0.0, 1.0 - progress / 0.25)

    def _inject_privileged(self, sequences, action_mask, attention_mask,
                           old_values=None, returns=None, inject_prob=0.0):
        device = sequences.device
        dtype = attention_mask.dtype
        batch_size = sequences.shape[0]

        self._inject_total += batch_size

        can_inject = (inject_prob > 0.0
                      and self.answer_map and len(self.answer_map) > 0
                      and self._tokenizer_for_answer is not None)

        if attention_mask.shape[1] < sequences.shape[1]:
            _extra = sequences.shape[1] - attention_mask.shape[1]
            attention_mask = torch.nn.functional.pad(attention_mask, (0, _extra), value=0)

        if can_inject:
            min_len = min(sequences.shape[1], attention_mask.shape[1], action_mask.shape[1])
            if old_values is not None:
                min_len = min(min_len, old_values.shape[1], returns.shape[1])
            if sequences.shape[1] > min_len:
                sequences = sequences[:, :min_len]
            if attention_mask.shape[1] > min_len:
                attention_mask = attention_mask[:, :min_len]
            if action_mask.shape[1] > min_len:
                action_mask = action_mask[:, :min_len]
            if old_values is not None:
                if old_values.shape[1] > min_len:
                    old_values = old_values[:, :min_len]
                if returns.shape[1] > min_len:
                    returns = returns[:, :min_len]

        fs_offset = 0
        pre_len = sequences.shape[1]
        prompt_ends = None
        ans_lens = None
        inject_flags = None

        if not can_inject:
            return (sequences, attention_mask, action_mask, old_values, returns,
                    fs_offset, prompt_ends, ans_lens, inject_flags, pre_len)

        inject_mask = torch.rand(batch_size, device=device) < inject_prob
        inject_idx = inject_mask.nonzero(as_tuple=False).view(-1).tolist()

        prompt_ends = [0] * batch_size
        for i in range(batch_size):
            act = action_mask[i]
            first_action = (act == 1).nonzero(as_tuple=True)[0]
            if first_action.numel() != 0:
                prompt_ends[i] = first_action[0].item()

        if self.few_shot_examples is not None and len(self.few_shot_examples) > 0:
            n_examples = min(2, len(self.few_shot_examples))
            fs = torch.cat([t.to(device) for t in self.few_shot_examples[:n_examples]], dim=0)
            fs_offset = fs.shape[0]

        ans_lens = [0] * batch_size
        inject_flags = [False] * batch_size

        new_seqs = [sequences[i] for i in range(batch_size)]
        new_attn = [attention_mask[i] for i in range(batch_size)]
        new_act = [action_mask[i] for i in range(batch_size)]
        new_vals = [old_values[i] for i in range(batch_size)] if old_values is not None else None
        new_rets = [returns[i] for i in range(batch_size)] if returns is not None else None

        for i in inject_idx:
            q_len = prompt_ends[i]
            if q_len <= 0:
                continue

            prompt_tokens = sequences[i, :q_len]
            prompt_text = self._tokenizer_for_answer.decode(prompt_tokens, skip_special_tokens=True).strip()
            self._inject_attempted += 1
            ans = self.answer_map.get(prompt_text, "")
            if not ans:
                for k, v in self.answer_map.items():
                    if (prompt_text.startswith(k[:80]) or k.startswith(prompt_text[:80])
                            or (len(k) > 20 and (k in prompt_text or prompt_text in k))):
                        ans = v
                        break
            if ans:
                self._inject_hit += 1
                ans_ids = self._tokenizer_for_answer.encode(f"A: {ans}\n", add_special_tokens=False)
                ans_tok = torch.tensor(ans_ids, dtype=torch.long, device=device)
                alen = ans_tok.numel()
            else:
                ans_tok = torch.tensor([], dtype=torch.long, device=device)
                alen = 0
            ans_lens[i] = alen
            inject_flags[i] = True

            prefix = sequences[i, :q_len]
            suffix = sequences[i, q_len:]
            if fs_offset > 0:
                seq_new = torch.cat([fs, prefix, ans_tok, suffix])
            else:
                seq_new = torch.cat([prefix, ans_tok, suffix])
            new_seqs[i] = seq_new

            attn_mid = torch.ones(alen, dtype=dtype, device=device)
            attn_new = torch.cat([attention_mask[i, :q_len], attn_mid, attention_mask[i, q_len:]])
            if fs_offset > 0:
                attn_new = torch.cat([torch.ones(fs_offset, dtype=dtype, device=device), attn_new])
            new_attn[i] = attn_new

            act_mid = torch.zeros(alen, dtype=action_mask.dtype, device=device)
            act_new = torch.cat([action_mask[i, :q_len], act_mid, action_mask[i, q_len:]])
            if fs_offset > 0:
                act_new = torch.cat([torch.zeros(fs_offset, dtype=action_mask.dtype, device=device), act_new])
            new_act[i] = act_new

            if old_values is not None:
                val_mid = torch.zeros(alen, dtype=old_values.dtype, device=device)
                val_new = torch.cat([old_values[i, :q_len], val_mid, old_values[i, q_len:]])
                if fs_offset > 0:
                    val_new = torch.cat([torch.zeros(fs_offset, dtype=old_values.dtype, device=device), val_new])
                new_vals[i] = val_new

                ret_mid = torch.zeros(alen, dtype=returns.dtype, device=device)
                ret_new = torch.cat([returns[i, :q_len], ret_mid, returns[i, q_len:]])
                if fs_offset > 0:
                    ret_new = torch.cat([torch.zeros(fs_offset, dtype=returns.dtype, device=device), ret_new])
                new_rets[i] = ret_new

        max_len = max(s.shape[0] for s in new_seqs)
        def pad_to(tensors, max_l, val=0):
            out = torch.full((batch_size, max_l), val, dtype=tensors[0].dtype, device=device)
            for i, t in enumerate(tensors):
                out[i, :t.shape[0]] = t
            return out

        sequences = pad_to(new_seqs, max_len, 0)
        attention_mask = pad_to(new_attn, max_len, 1)
        action_mask = pad_to(new_act, max_len, 0)
        if old_values is not None:
            old_values = pad_to(new_vals, max_len, 0)
            returns = pad_to(new_rets, max_len, 0)

        global_max = torch.tensor([max_len], device=device, dtype=torch.long)
        torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX)
        global_max = int(global_max.item())
        if global_max > max_len:
            extra = global_max - max_len
            sequences = torch.nn.functional.pad(sequences, (0, extra), value=0)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, extra), value=1)
            action_mask = torch.nn.functional.pad(action_mask, (0, extra), value=0)
            if old_values is not None:
                old_values = torch.nn.functional.pad(old_values, (0, extra), value=0)
                returns = torch.nn.functional.pad(returns, (0, extra), value=0)

        return (sequences, attention_mask, action_mask, old_values, returns,
                fs_offset, prompt_ends, ans_lens, inject_flags, pre_len)

    def ppo_train(self):
        if self.args.train.dynamic_batch_enable:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        should_shuffle = (
            self.strategy.ring_attn_group is None
            and self.args.ds.tensor_parallel_size <= 1
            and not self.args.train.dynamic_batch_enable
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=should_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        adaptive_enabled = getattr(self.args, "adaptive_critic", False)

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for step, experience in enumerate(pbar):
                experience.to_device(device)

                if adaptive_enabled:
                    step_kl = _compute_step_kl(experience)
                    step_accuracy = _extract_step_accuracy(experience)
                    step_entropy = _compute_step_entropy(experience)

                    signals = torch.tensor(
                        [step_kl, step_accuracy, step_entropy],
                        device=device, dtype=torch.float32,
                    )
                    torch.distributed.all_reduce(signals, op=torch.distributed.ReduceOp.AVG)
                    step_kl = signals[0].item()
                    step_accuracy = signals[1].item()
                    step_entropy = signals[2].item()

                    decision = self._step_adaptive_decision(step_kl, step_accuracy, step_entropy)
                    step_update = decision["critic_updated"]
                else:
                    step_update = True
                    decision = {}

                status = self.training_step(experience, step, critic_update_enabled=step_update)

                if adaptive_enabled:
                    status["critic_step_updated"] = 1.0 if step_update else 0.0
                    status["critic_step_kl"] = decision.get("kl", 0.0)
                    status["critic_step_accuracy"] = decision.get("accuracy", 0.0)
                    status["critic_step_acc_growth"] = decision.get("acc_growth", -1.0)
                    status["critic_step_entropy"] = decision.get("entropy", 0.0)
                    status["critic_step_entropy_ema"] = decision.get("entropy_ema", 0.0)
                    status["critic_step_entropy_collapse"] = decision.get("entropy_collapse", 0.0)
                    status["critic_step_three_track"] = 1.0 if decision.get("three_track_trigger") else 0.0
                    status["critic_step_reason_id"] = float(decision.get("reason_id", 0))
                    status["critic_step_total_updates"] = float(decision.get("total_updates", 0))
                    status["critic_step_total_steps"] = float(decision.get("total_steps", 0))
                    status["critic_step_consecutive_skips"] = float(decision.get("consecutive_skips", 0))

                status = self.strategy.all_reduce(status)

                status["skip%"] = (1 - self._total_updates / max(self._total_steps, 1)) * 100

                status_list.append(status)
                pbar.set_postfix(status)

        if status_list:
            status_mean = dict(status_list[0])
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean, status_list

    def training_step(self, experience: Experience, step: int, critic_update_enabled: bool = True) -> Dict[str, float]:
        self.critic.train()

        sequences = experience.sequences
        old_values = experience.values
        returns = experience.returns
        action_mask = experience.action_mask
        packed_seq_lens = None
        attention_mask = experience.attention_mask

        min_len = min(sequences.shape[1], action_mask.shape[1],
                      attention_mask.shape[1], old_values.shape[1],
                      returns.shape[1])
        if sequences.shape[1] > min_len:
            sequences = sequences[:, :min_len]
        if action_mask.shape[1] > min_len:
            action_mask = action_mask[:, :min_len]
        if attention_mask.shape[1] > min_len:
            attention_mask = attention_mask[:, :min_len]
        if old_values.shape[1] > min_len:
            old_values = old_values[:, :min_len]
        if returns.shape[1] > min_len:
            returns = returns[:, :min_len]

        if self._privilege_prob_enabled:
            inject_prob = self._inject_prob("train")
        else:
            inject_prob = 0.0
        sequences, attention_mask, action_mask, old_values, returns, _, _, _, _, _ = \
            self._inject_privileged(
                sequences, action_mask, attention_mask,
                old_values=old_values, returns=returns, inject_prob=inject_prob,
            )

        loss_batch_info = get_loss_batch_info(
            self.strategy,
            action_mask,    # use extended action_mask
            replay_buffer=self.replay_buffer,
            step=step,
            dynamic_batch=self.args.train.dynamic_batch_enable,
        )

        # ── The critic's value head produces values 1 token shorter than the
        #     input (BOS/EOS handling in get_llm_for_sequence_regression).
        #     Trim the last token from action_mask to match.  This is safe
        #     because the last position is always padding (never an action).
        action_mask = action_mask[:, :-1]
        old_values = old_values[:, :-1]
        returns = returns[:, :-1]

        values, output = self.critic(
            sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            values_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )

        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=action_mask,
            **loss_batch_info,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        aux_loss = aux_loss * self.args.actor.aux_loss_coef
        if self.args.train.dynamic_batch_enable:
            aux_loss = aux_loss * self.replay_buffer.dynamic_sample_loss_scale[step]
        loss = critic_loss + aux_loss

        critic_grad_norm = 0.0
        if critic_update_enabled:
            self.strategy.backward(loss, self.critic, self.critic_optim)
            if self.args.train.dynamic_batch_enable:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
            else:
                self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
            critic_grad_norm = self.strategy.get_grad_norm(self.critic)

        # status
        status = {
            "critic_loss": critic_loss.detach().item(),
            "values": masked_mean(values, action_mask).detach().item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
            "critic_grad_norm": critic_grad_norm,
        }
        return status


@ray.remote(num_gpus=1)
class CriticModelActorStar(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps):
        args = strategy.args
        self.disable_ds_ckpt = args.ckpt.disable_ds

        self._setup_distributed(strategy)
        if os.environ.get("PPO_CRITIC_W_OFFLOAD", "1") == "1":
            _orig_get_ds_train = strategy.get_ds_train_config

            def _patched_get_ds_train(*args, **kwargs):
                cfg = _orig_get_ds_train(*args, **kwargs)
                zo = cfg.setdefault("zero_optimization", {})
                zo.setdefault("offload_param", {})["device"] = "cpu"
                zo.setdefault("offload_optimizer", {})["device"] = "cpu"
                return cfg

            strategy.get_ds_train_config = _patched_get_ds_train
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.reward.normalize_enable,
            attn_implementation=strategy.args.ds.attn_implementation,
            experts_implementation=strategy.args.ds.experts_implementation,
            param_dtype=strategy.args.ds.param_dtype,  # default: bf16
            load_in_4bit=strategy.args.ds.load_in_4bit,
            lora_rank=strategy.args.ds.lora.rank,
            lora_alpha=strategy.args.ds.lora.alpha,
            target_modules=strategy.args.ds.lora.target_modules,
            lora_dropout=strategy.args.ds.lora.dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=strategy.args.ds.value_head_prefix,
            init_value_head=strategy.args.actor.model_name_or_path == strategy.args.critic.model_name_or_path,
            packing_samples=strategy.args.ds.packing_samples,
        )
        strategy.print(critic)
        strategy.print("reward normalization status: {}".format(strategy.args.reward.normalize_enable))
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

        # configure tokenizer (CriticStar always needs it for few-shot encoding)
        self.tokenizer = get_tokenizer(
            pretrain, critic, "left", strategy, use_fast=not strategy.args.data.disable_fast_tokenizer
        )

        if args.actor.gradient_checkpointing_enable:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.actor.gradient_checkpointing_reentrant}
            )
        if os.environ.get("PPO_DS_ACT_OFFLOAD", "0") == "1":
            _patch_ds_config_cpu_checkpointing()

        # Critic reads its own args.critic.* sub-namespace.  Typical setup: actor may
        # use Muon but --critic.optim stays adam because value heads are essentially 1D.
        critic_cfg = dict(
            optim=args.critic.optim,
            muon=vars(args.critic.muon),
            adam=vars(args.critic.adam),
            lr_scheduler=args.critic.lr_scheduler,
            lr_warmup_ratio=args.critic.lr_warmup_ratio,
            min_lr_ratio=args.critic.min_lr_ratio,
            max_norm=args.critic.max_norm,
            scheduler_steps=max_steps,
        )
        self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare((critic, critic_cfg))

        if os.environ.get("PPO_PREWARM_ADAM", "1") == "1":
            _prewarm_adam_states(self.critic_optim)

        if os.environ.get("PPO_DS_ACT_OFFLOAD", "0") == "1":
            _replace_checkpoint_func(self.critic)

        # load checkpoint
        ckpt_path = os.path.join(args.ckpt.path, "_critic")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.critic, ckpt_path)

        # initial offload
        if strategy.args.ds.enable_sleep:
            self.offload_states()

        # ── Prepare few-shot examples (fixed 2-shot) ──
        few_shot_text = os.environ.get("PPO_FEWSHOT_TEXT", "").strip()
        few_shot_examples = None
        if few_shot_text:
            # Split into individual examples (separated by double newline)
            raw_examples = [e.strip() for e in few_shot_text.split("\n\n") if e.strip()]
            few_shot_examples = []
            for ex in raw_examples:
                ids = self.tokenizer.encode(ex, add_special_tokens=False)
                few_shot_examples.append(torch.tensor(ids, dtype=torch.long))
            strategy.print(f"[CriticStar] {len(few_shot_examples)} few-shot examples prepared")

        # ── Prepare answer map for privileged (q, a) conditioning ──
        answer_map_path = os.environ.get("PPO_ANSWER_MAP", "").strip()
        if answer_map_path:
            strategy.print(f"[AdaptivePPO*] Answer map path: {answer_map_path}")

        self.trainer = CriticPPOTrainerStar(
            strategy,
            critic=self.critic,
            critic_optim=self.critic_optim,
            critic_scheduler=self.critic_scheduler,
            micro_train_batch_size=args.train.micro_batch_size,
            value_clip=args.critic.value_clip,
            few_shot_examples=few_shot_examples,
            total_train_steps=max_steps,
            answer_map_path=answer_map_path,
        )
        self.trainer._tokenizer_for_answer = self.tokenizer
        self.critic_episode_count = 0
        self._last_saved_grad_step = 0

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.critic.eval()

        orig_len = sequences.shape[1]
        orig_am_len = action_mask.shape[1]
        if self.trainer._privilege_prob_enabled:
            inject_prob = self.trainer._inject_prob("rollout")
        else:
            inject_prob = 0.0
        sequences, action_mask, attention_mask, _, _, fs_off, prompt_ends, ans_lens, inject_flags, pre_len = \
            self.trainer._inject_privileged(
                sequences.to(device), action_mask.to(device), attention_mask.to(device),
                old_values=None, returns=None, inject_prob=inject_prob,
            )
        if prompt_ends is not None:
            action_mask = action_mask[:, :-1]

        if action_mask.shape[1] > sequences.shape[1] - 1:
            action_mask = action_mask[:, :sequences.shape[1] - 1]

        if attention_mask.shape[1] != sequences.shape[1]:
            if attention_mask.shape[1] < sequences.shape[1]:
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, sequences.shape[1] - attention_mask.shape[1]), value=0)
            else:
                attention_mask = attention_mask[:, :sequences.shape[1]]

        if os.environ.get("PPO_DEBUG_SHAPES", "0") == "1":
            print(f"[CRITIC] fwd shapes: seq={tuple(sequences.shape)} "
                  f"act={tuple(action_mask.shape)} attn={tuple(attention_mask.shape)} "
                  f"prompt_ends={'yes' if prompt_ends is not None else 'no'}")

        with torch.no_grad():
            values_full = self.critic(
                sequences,
                action_mask,
                attention_mask,
                ring_attn_group=self.strategy.ring_attn_group,
                values_allgather=True,
            )

        if prompt_ends is None:
            out = values_full
            if out.shape[1] < orig_am_len:
                out = torch.nn.functional.pad(out, (0, orig_am_len - out.shape[1]), value=0)
        else:
            batch = sequences.shape[0]
            out = torch.zeros(batch, orig_len - 1, dtype=values_full.dtype, device=values_full.device)
            for i in range(batch):
                q_len = prompt_ends[i]
                if inject_flags[i]:
                    r_start = fs_off + q_len + ans_lens[i]
                else:
                    r_start = q_len
                r_vals = pre_len - q_len - 1
                out[i, q_len:q_len + r_vals] = values_full[i, r_start:r_start + r_vals]

        self.critic.train()
        return out.to("cpu")

    def append(self, experience):
        """Append experience to replay buffer."""
        self.trainer.replay_buffer.append(experience)

    def fit(self):
        """Train critic model with the replay buffer.

        Per-step adaptive decisions happen inside ppo_train().
        No more episode-level SKIP/TRAIN branching — every episode always
        enters training, but individual steps may skip backward when the
        three-track AND gate + safety nets determine the critic is healthy.
        """
        torch.cuda.empty_cache()
        self.critic.train()

        self.critic_episode_count += 1
        current_ep = self.critic_episode_count

        adaptive_enabled = getattr(self.strategy.args, "adaptive_critic", False)

        if os.environ.get("PPO_ACT_OFFLOAD", "0") == "1":
            with _activation_offload_ctx():
                status, step_statuses = self.trainer.ppo_train()
        else:
            status, step_statuses = self.trainer.ppo_train()

        total_steps = self.trainer._total_steps
        total_updates = self.trainer._total_updates
        update_ratio = total_updates / max(total_steps, 1)

        status.update({
            "critic_updated": True,
            "total_critic_updates": total_updates,
            "total_critic_steps": total_steps,
            "critic_update_ratio": update_ratio,
        })
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        _save_every = int(getattr(self.strategy.args.ckpt, "save_every_grad_steps", 0) or 0)
        if _save_every > 0:
            _steps_sync = total_steps
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                _t = torch.tensor([float(total_steps)], device=torch.cuda.current_device())
                torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MAX)
                _steps_sync = int(_t.item())
            if _steps_sync // _save_every > self._last_saved_grad_step // _save_every:
                self.save_checkpoint(f"gradstep{_steps_sync}")
                self._last_saved_grad_step = _steps_sync

        if not self.strategy.is_rank_0():
            return status

        if adaptive_enabled and step_statuses:
            n_updated = sum(1 for ss in step_statuses if ss.get("critic_step_updated", 0) > 0.5)
            n_skipped = len(step_statuses) - n_updated
            reasons = []
            for ss in step_statuses:
                rid = int(ss.get("critic_step_reason_id", 0))
                if rid > 0:
                    reason_map = {1: "three_track", 2: "kl_ceiling", 3: "max_skip", 4: "kl_delta", 5: "ent_low"}
                    reasons.append(reason_map.get(rid, "unknown"))
        else:
            n_updated = len(step_statuses)
            reasons = []

        if step_statuses:
            for i, ss in enumerate(step_statuses):
                step_entry = {}
                for k, v in ss.items():
                    step_entry[f"critic_{k}"] = _safe_float(v)
                step_entry["step"] = "critic_step"
                step_entry["episode"] = current_ep
                step_entry["batch"] = i + 1
                _append_metrics(step_entry)

        ep_entry = {f"critic_{k}": v for k, v in status.items()}
        ep_entry["step"] = "critic_episode"
        ep_entry["episode"] = current_ep
        if adaptive_enabled:
            inj_h = self.trainer._inject_hit
            inj_a = self.trainer._inject_attempted
            inj_t = self.trainer._inject_total
            print(f"[CRITIC] >>> DONE | loss={_safe_float(status.get('critic_loss', 0)):.4f} "
                  f"| values={_safe_float(status.get('values', 0)):.4f} "
                  f"| step_updates={n_updated}/{len(step_statuses)} "
                  f"({n_updated/max(len(step_statuses),1)*100:.0f}%) "
                  f"| cumulative={total_updates}/{total_steps} "
                  f"({update_ratio*100:.1f}%) "
                  f"| inject={inj_a}/{inj_t} ({inj_a/max(inj_t,1)*100:.0f}%) "
                  f"hit={inj_h}/{inj_a} ({inj_h/max(inj_a,1)*100:.0f}%) "
                  f"| reasons={reasons[:5]}{'...' if len(reasons)>5 else ''}")
        else:
            print(f"[CRITIC] >>> TRAINED | loss={_safe_float(status.get('critic_loss', 0)):.4f} "
                  f"| values={_safe_float(status.get('values', 0)):.4f}")
        _append_metrics(ep_entry)

        return status

    def save_model(self):
        args = self.strategy.args
        if self.tokenizer is None:
            return

        self.strategy.save_model(
            self.critic,
            self.tokenizer,
            args.ckpt.output_dir + "_critic",
        )

    def save_checkpoint(self, tag, metric_value=None, metric_key=None):
        args = self.strategy.args
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.critic,
                os.path.join(args.ckpt.path, "_critic"),
                tag,
                args.ckpt.max_num,
                args.ckpt.max_mem,
                metric_value=metric_value,
                metric_key=metric_key,
            )

        if (getattr(args.critic, "save_value_network", False)
                and self.tokenizer is not None):
            hf_root = str(args.ckpt.path).rstrip("/\\") + "_critic_hf"
            hf_dir = os.path.join(hf_root, str(tag))
            os.makedirs(hf_root, exist_ok=True)
            self.strategy.save_model(self.critic, self.tokenizer, hf_dir)
            _prune_hf_ckpts(hf_root, int(getattr(args.ckpt, "max_num", 3) or 3))

    def reload_states(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        reload_deepspeed_states(self.critic)

    def offload_states(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        offload_deepspeed_states(self.critic)
