import os
import json
import time
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


def _append_metrics(entry: dict) -> None:
    lock_path = _METRICS_PATH + ".lock"
    fd = None
    for _ in range(200):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL)
            break
        except OSError:
            time.sleep(0.01)
    try:
        with open(_METRICS_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()
    except Exception as e:
        print(f"[ERROR] Failed to write metrics to {_METRICS_PATH}: {e}")
    finally:
        if fd is not None:
            os.close(fd)
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


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

class CriticPPOTrainer(ABC):
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

        # ── Per-step Adaptive Critic state (replaces episode-level _read_metrics_state) ──
        self._consecutive_skips: int = 0
        self._prev_kl: float = 0.0
        self._total_steps: int = 0
        self._total_updates: int = 0
        self._entropy_ema: float = 0.0
        self._acc_ema: float = 0.0
        self._acc_history: list[float] = []

    def _step_adaptive_decision(self, step_kl: float, step_accuracy: float, step_entropy: float) -> dict:
        """Per-step three-track AND gate + safety nets.

        Mirrors the logic of _build_adaptive_status, but operates on per-mini-batch
        signals computed directly from Experience instead of reading episode-level
        actor metrics from the JSONL file.
        """
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

        # ── Thresholds ──
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

        acc_surge_th = getattr(args, "adaptive_acc_surge_threshold", 0.0)
        if acc_surge_th <= 0.0:
            _env_surge = os.environ.get("PPO_ACC_SURGE_THRESHOLD", "0")
            acc_surge_th = float(_env_surge) if _env_surge not in ("", "none", "None") else 0.0
        if not critic_updated and acc_surge_th > 0 and acc_growth > acc_surge_th:
            critic_updated = True
            reason = "acc_surge"
            reason_id = 7

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

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
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

                # ── Per-step adaptive decision ──
                if adaptive_enabled:
                    step_kl = _compute_step_kl(experience)
                    step_accuracy = _extract_step_accuracy(experience)
                    step_entropy = _compute_step_entropy(experience)

                    # All-reduce signals across DP ranks for a unified decision.
                    # Without this, different ranks may see different local mini-batches
                    # and make conflicting step_update decisions, causing NCCL timeout
                    # (some ranks enter backward while others skip the collective).
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

                # Merge per-step adaptive signals into status (all floats for all_reduce safety)
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

                # for DP
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
        loss_batch_info = get_loss_batch_info(
            self.strategy,
            action_mask,
            replay_buffer=self.replay_buffer,
            step=step,
            dynamic_batch=self.args.train.dynamic_batch_enable,
        )

        # critic loss
        values, output = self.critic(
            sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            values_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )

        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
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

        status = {
            "critic_loss": critic_loss.detach().item(),
            "values": masked_mean(values, experience.action_mask).detach().item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
            "critic_grad_norm": critic_grad_norm,
        }
        return status


@ray.remote(num_gpus=1)
class CriticModelActor(BaseModelActor):
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

        self.tokenizer = None
        if strategy.args.critic.save_value_network:
            self.tokenizer = get_tokenizer(
                pretrain, critic, "left", strategy, use_fast=not strategy.args.data.disable_fast_tokenizer
            )

        if args.actor.gradient_checkpointing_enable:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.actor.gradient_checkpointing_reentrant}
            )

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
        ckpt_path = os.path.join(args.ckpt.path, "_critic")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.critic, ckpt_path)

        if strategy.args.ds.enable_sleep:
            self.offload_states()

        self.trainer = CriticPPOTrainer(
            strategy,
            critic=self.critic,
            critic_optim=self.critic_optim,
            critic_scheduler=self.critic_scheduler,
            micro_train_batch_size=args.train.micro_batch_size,
            value_clip=args.critic.value_clip,
        )
        self.critic_episode_count = 0
        self._last_saved_grad_step = 0

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                values_allgather=True,
            )
        self.critic.train()  # reset model state
        return value.to("cpu")

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
                    reason_map = {1: "three_track", 2: "kl_ceiling", 3: "max_skip", 4: "kl_delta", 5: "ent_low", 7: "acc_surge"}
                    reasons.append(reason_map.get(rid, "unknown"))
        else:
            n_updated = len(step_statuses)
            n_skipped = 0
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
            print(f"[CRITIC] >>> DONE | loss={_safe_float(status.get('critic_loss', 0)):.4f} "
                  f"| values={_safe_float(status.get('values', 0)):.4f} "
                  f"| step_updates={n_updated}/{len(step_statuses)} "
                  f"({n_updated/max(len(step_statuses),1)*100:.0f}%) "
                  f"| cumulative={total_updates}/{total_steps} "
                  f"({update_ratio*100:.1f}%) "
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
