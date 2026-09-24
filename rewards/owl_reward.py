"""
OWL RLVR AgentExecutor — IT Operations Knowledge MCQ Reward.

Extracts answer letter (a/b/c/d) from model response.
  answer = a/b/c/d, reward = 1.0 if correct, 0.0 otherwise.

Usage:
  --train.agent_func_path=/path/to/owl_reward.py
"""

import re
from copy import deepcopy
from typing import Optional

from openrlhf.utils.agent import AgentExecutorBase


# ══════════════════════════════════════════════════════════════════════════════
# Answer extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_answer(text: str) -> Optional[str]:
    """
    Extract predicted answer letter (a/b/c/d) from model response.
    Only explicit "answer: X" / "correct: X" patterns — no bare letter fallback.
    """
    if not text:
        return None

    text_lower = text.lower().strip()
    lines = [l.strip() for l in text_lower.split("\n") if l.strip()]

    # Search from the last line: only explicit "answer: X" patterns
    for line in reversed(lines):
        patterns = [
            r"(?:answer|correct|choice|option|选|therefore|所以|thus)[：:\s]*([a-d])",
            r"answer\s+is\s+\(?([a-d])\)?",
            r"option\s+\(?([a-d])\)?",
            r"is\s+correct\s*\(?([a-d])\)?",
        ]
        for p in patterns:
            m = re.search(p, line)
            if m and m.group(1) in ("a", "b", "c", "d"):
                return m.group(1)

    return None


def _parse_label(label) -> Optional[str]:
    """Parse ground-truth label: a/b/c/d. Also handles "D." format."""
    if label is None:
        return None
    s = str(label).strip().lower().replace(".", "")
    if s in ("a", "b", "c", "d"):
        return s
    return None


# ══════════════════════════════════════════════════════════════════════════════
# AgentExecutor
# ══════════════════════════════════════════════════════════════════════════════

class AgentExecutor(AgentExecutorBase):
    """
    Single-turn agent that answers an IT operations MCQ and scores it
    against ground-truth label (a/b/c/d).
    """

    async def execute(self, prompt, label, sampling_params, max_length,
                      hf_tokenizer, llm_engine, images=None):
        try:
            return await self._execute(prompt, label, sampling_params,
                                       max_length, hf_tokenizer, llm_engine, images)
        except Exception:
            import traceback
            traceback.print_exc()
            prompt_token_ids = hf_tokenizer(
                text=prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].tolist()
            return {
                "prompt": prompt, "label": label, "images": images,
                "mm_train_inputs": None,
                "observation_tokens": prompt_token_ids,
                "action_ranges": [(0, len(prompt_token_ids))],
                "rollout_log_probs": None,
                "truncated": False,
                "reward": 0.0, "scores": 0.0,
                "extra_logs": {},
            }

    async def _execute(self, prompt, label, sampling_params, max_length,
                       hf_tokenizer, llm_engine, images=None):
        prompt_token_ids = hf_tokenizer(
            text=prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()

        effective_params = sampling_params
        if sampling_params.max_tokens is None:
            effective_params = deepcopy(sampling_params)
            effective_params.max_tokens = max_length - len(prompt_token_ids)

        request_output = await llm_engine.generate(
            prompt_token_ids, deepcopy(effective_params)
        )
        generation_output = request_output.outputs[0]
        action_token_ids = generation_output.token_ids
        is_truncated = generation_output.finish_reason == "length"

        observation_token_ids = prompt_token_ids + action_token_ids
        action_ranges = [(len(prompt_token_ids), len(observation_token_ids))]

        rollout_log_probs = None
        if sampling_params.logprobs is not None and generation_output.logprobs is not None:
            rollout_log_probs = [0.0] * len(prompt_token_ids)
            for logprob in generation_output.logprobs:
                rollout_log_probs.append(logprob.logprob)

        response_text = hf_tokenizer.decode(action_token_ids, skip_special_tokens=True)

        pred = extract_answer(response_text)
        gt = _parse_label(label)

        if pred is not None and gt is not None and pred == gt:
            reward = 1.0
        else:
            reward = 0.0

        return {
            "prompt": prompt, "label": label, "images": images,
            "mm_train_inputs": None,
            "observation_tokens": observation_token_ids,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            "truncated": is_truncated,
            "reward": reward, "scores": reward,
            "extra_logs": {},
        }
