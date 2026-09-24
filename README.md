# AdaPPO — Anonymous Code Release

Code for **"AdaPPO: Resolving Dynamics Asymmetry in LLM Alignment via Adaptive Gating"**.

AdaPPO schedules *critic* updates adaptively. Instead of updating the critic at every step, it
refreshes it only when a joint three-gate condition fires — policy performance ∧ policy drift (KL)
∧ entropy collapse — and otherwise freezes the critic while still running the forward pass that
produces the value estimates used for advantage estimation. A hard freeze ceiling plus three safety
guards (KL ceiling, KL surge, entropy floor) bound staleness. **AdaPPO\*** additionally injects
privileged information into the critic.

This repository contains **only the files we changed or added on top of OpenRLHF** — the actor,
critic and entry-point modifications — together with the launch scripts and the rule-based reward
functions used in the paper.

## Repository layout

| Path | What it is |
|---|---|
| `openrlhf_patch/trainer/ray/adappo_critic.py` | **AdaPPO core** — adaptive critic scheduling (three gates + freeze ceiling + safety net) |
| `openrlhf_patch/trainer/ray/adappo_star_critic.py` | **AdaPPO\*** critic — privileged-information (PI) injection |
| `openrlhf_patch/trainer/ray/adappo_actor.py` | actor-side changes |
| `openrlhf_patch/cli/train_ppo_ray.py` | entry point — adds the `--adaptive_*` flags (everything else is upstream) |
| `scripts/run_*.sh` | per-dataset launch commands with the paper hyperparameters |
| `rewards/*_reward.py` | deterministic rule-based rewards, passed via `--train.agent_func_path` |

## Environment

Single server, 8 × A100-80G, bf16, DeepSpeed ZeRO-3, Ray + vLLM rollout, PyTorch + transformers +
deepspeed + vllm.

## Install

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF && pip install -e .
cp -r <ANON_REPO>/openrlhf_patch/* openrlhf/      # only change needed
```

The patch mirrors the OpenRLHF package layout and imports internal modules, so use the revision
pinned in `requirements.txt`. The entry point imports `adappo_actor.py` / `adappo_critic.py` /
`adappo_star_critic.py`; upstream `ppo_critic.py` is untouched.

## Quick start

```bash
export OPENRLHF_ROOT=/path/to/OpenRLHF
bash scripts/run_gsm8k_adappo.sh      # GSM8K
bash scripts/run_mbpp_adappo.sh       # ... mbpp / logiqa / owl, same pattern
```

Each script calls `python -m openrlhf.cli.train_ppo_ray` with the paper hyperparameters (Ray is
initialised by the entry point itself); per-step metrics go to `runs/ppo_metrics.jsonl`. The same
scripts launch **AdaPPO\*** — see below.

## AdaPPO vs AdaPPO\*

AdaPPO\* replaces the critic module with `adappo_star_critic.py`, which additionally injects
privileged information (the reference answer) into the critic's input; the actor, the rollout
sampling distribution, the reward function and the gating logic are unchanged. To run it, launch the
same `scripts/run_<dataset>_adappo.sh` and (i) uncomment the star import at the top of
`openrlhf/cli/train_ppo_ray.py`, then (ii) point `PPO_ANSWER_MAP` (JSONL of `{"question","answer"}`)
and `PPO_FEWSHOT_TEXT` at your data.

## Hyperparameters

We train on GSM8K / MBPP / LogiQA / OWL with 1–4 episodes, rollout batch 16–32, micro batch 2–4,
and KL coefficient 0.15–0.30. The three gating thresholds (ΔPerf, KL, entropy) are set
per dataset, with a hard freeze ceiling of 6 steps and three safety guards (KL ceiling, KL surge,
entropy floor). Full configurations — including the exact per-dataset
values — are given in `scripts/` and the supplementary material.

The gate signals are computed every PPO step: ΔPerf (EMA-smoothed accuracy slope over a 5-step
window), mean KL between the current and the behaviour policy, and raw token entropy.

## Metrics

Per-step metrics (accuracy, reward, KL, entropy, whether the critic was updated, the gate states
and the freeze counter) are written to `runs/ppo_metrics.jsonl` (override with `PPO_METRICS_FILE`).

## Data

Prepare the datasets locally and point `--data.prompt_dataset` at them (keys in parentheses):

| Dataset | Keys | Reward |
|---|---|---|
| GSM8K | `question`, `answer` | exact match of the final numeric answer |
| MBPP | `question`, `answer` | unit-test pass rate |
| LogiQA | `question`, `answer` | option matching |
| OWL | `question`, `answer` | option matching |

## Citation

```bibtex
@misc{adappo_anonymous,
  title  = {AdaPPO: Resolving Dynamics Asymmetry in LLM Alignment via Adaptive Gating},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## License and acknowledgement

MIT (see `LICENSE`). The files under `openrlhf_patch/` are modified copies of
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and remain under its Apache-2.0 license.
We thank the OpenRLHF authors for the training infrastructure.
