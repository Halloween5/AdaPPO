import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from typing import Optional

from openrlhf.utils.agent import AgentExecutorBase

def extract_code(text: str, is_truncated: bool = False,
                 target_func: Optional[str] = None) -> Optional[str]:
    if not text:
        return None

    if "</think" in text:
        text = re.split(r"</think\w*>", text)[-1]
        if not text.strip():
            return None

    def _clean(code: str) -> str:
        code = re.sub(r"\)(def |class )", r")\n\1", code)
        return code.strip()

    def _preceding_imports(src: str, start: int) -> list:
        pre: list = []
        for l in reversed(src[:start].split("\n")):
            s = l.rstrip()
            if s.startswith(("import ", "from ")):
                pre.insert(0, s)
            elif s.strip() == "":
                continue
            else:
                break
        return pre

    def _collect_body(src: str, start: int) -> str:
        seg_lines = src[start:].split("\n")
        code = [seg_lines[0]]
        for l in seg_lines[1:]:
            if l.strip() and not l.startswith((" ", "\t")):
                break
            code.append(l)
        return _clean("\n".join(code))

    if target_func:
        pat = re.compile(r"def\s+" + re.escape(target_func) + r"\b")
        matches = list(re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL))
        for m in reversed(matches):
            if pat.search(m.group(1)):
                code = _clean(m.group(1))
                if code:
                    return code
        dms = list(pat.finditer(text))
        if dms:
            start = dms[-1].start()
            body = _collect_body(text, start)
            if body:
                pre = _preceding_imports(text, start)
                return ("\n".join(pre) + "\n" + body).strip() if pre else body

    matches = list(re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        code = _clean(matches[-1].group(1))
        if code:
            return code

    defs = list(re.finditer(r"^(def|class)\s+\w+", text, re.MULTILINE))
    if defs:
        start = defs[-1].start()
        body = _collect_body(text, start)
        if body:
            pre = _preceding_imports(text, start)
            return ("\n".join(pre) + "\n" + body).strip() if pre else body

    if is_truncated:
        m = re.search(r"```(?:python)?\s*\n(.+?)$", text, re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()

        lines = text.strip().split("\n")
        code_lines = []
        in_code = False
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                if in_code:
                    code_lines.insert(0, line)
                continue
            if in_code:
                if stripped.startswith("```"):
                    break
                code_lines.insert(0, line)
            elif stripped.startswith(("def ", "class ", "import ", "from ", "    ", "\t")):
                code_lines.insert(0, line)
                in_code = True
        if code_lines and len(code_lines) >= 2:
            return "\n".join(code_lines).strip()

    candidate = text.strip()
    if len(candidate) > 20:
        return candidate
    return None


def is_trivial_code(code: str) -> bool:
    if not code:
        return True
    lines = [l.strip() for l in code.split("\n") if l.strip()]
    if not lines:
        return True
    trivial_patterns = [
        r"^pass$", r"^return$", r"^return\s+None$", r"^return\s+0$",
        r"^return\s+\"\"$", r"^return\s+''$", r"^return\s+\[\]$",
        r"^\.\.\.$", r"^#.*$",
    ]
    trivial_count = sum(1 for line in lines if any(re.match(p, line) for p in trivial_patterns))
    return trivial_count / len(lines) > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def run_assert_tests(code: str, test_setup: str, test_list: list, timeout: float = 10.0) -> float:
    if not code or not test_list:
        return 0.0

    header = (test_setup or "").strip()
    tests = [str(t).strip() for t in test_list if str(t).strip()]
    if not tests:
        return 0.0

    test_body = "\n".join(
        f"try:\n    {t}\n    _results.append(1)\nexcept Exception:\n    _results.append(0)"
        for t in tests
    )
    parts = [p for p in (header, code, test_body, "print(sum(_results), len(_results))") if p.strip()]
    script = "\n\n".join(parts)

    try:
        compile(script, "<string>", "exec")
    except SyntaxError:
        return 0.0

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write("_results = []\n" + script)
            tmp_path = f.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return 0.0
        out = result.stdout.strip()
        try:
            passed_s, total_s = out.split()
            passed, total = int(passed_s), int(total_s)
        except Exception:
            return 0.0
        return passed / max(total, 1)
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def _parse_mbpp_label(label) -> tuple:
    try:
        data = json.loads(label) if isinstance(label, str) else label
        if isinstance(data, dict):
            return str(data.get("test_setup_code", "") or ""), list(data.get("test_list", []) or [])
        if isinstance(data, list):
            return "", [str(x) for x in data]
    except Exception:
        pass
    return "", []


def compute_mbpp_reward(code: Optional[str], test_setup: str, test_list: list,
                        is_truncated: bool) -> tuple:
    if not code:
        return 0.0, 0.0

    pass_rate = run_assert_tests(code, test_setup, test_list)

    compile_bonus = 0.05 if pass_rate > 0 else 0.0
    truncation_penalty = 0.20 if is_truncated else 0.0
    trivial_penalty = 0.20 if (pass_rate == 0 and is_trivial_code(code)) else 0.0

    repetition_penalty = 0.0
    def_names = re.findall(r"^def\s+(\w+)", code, re.MULTILINE)
    if def_names:
        most_common = Counter(def_names).most_common(1)[0]
        if most_common[1] > 2:
            repetition_penalty = 0.10

    raw = (pass_rate + compile_bonus - truncation_penalty
           - trivial_penalty - repetition_penalty)
    return max(0.0, min(1.0, raw)), pass_rate


# ══════════════════════════════════════════════════════════════════════════════
# AgentExecutor
# ══════════════════════════════════════════════════════════════════════════════

class AgentExecutor(AgentExecutorBase):

    async def execute(
        self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None,
    ):
        try:
            return await self._execute(prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images)
        except Exception:
            import traceback
            traceback.print_exc()
            prompt_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
            return {
                "prompt": prompt, "label": label, "images": images, "mm_train_inputs": None,
                "observation_tokens": prompt_ids, "action_ranges": [(0, len(prompt_ids))],
                "rollout_log_probs": None, "truncated": False, "reward": 0.0, "scores": 0.0, "extra_logs": {},
            }

    async def _execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None):
        prompt_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()

        effective_params = sampling_params
        if sampling_params.max_tokens is None:
            effective_params = deepcopy(sampling_params)
            effective_params.max_tokens = max(1, max_length - len(prompt_ids))

        max_prompt_len = max_length - effective_params.max_tokens
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        request_output = await llm_engine.generate(prompt_ids, deepcopy(effective_params))
        gen_output = request_output.outputs[0]
        action_ids = gen_output.token_ids
        is_truncated = gen_output.finish_reason == "length"

        obs_ids = prompt_ids + action_ids
        action_ranges = [(len(prompt_ids), len(obs_ids))]

        rollout_log_probs = None
        if sampling_params.logprobs is not None and gen_output.logprobs is not None:
            rollout_log_probs = [0.0] * len(prompt_ids)
            for tid, lp_dict in zip(action_ids, gen_output.logprobs):
                token_lp = lp_dict.get(tid)
                rollout_log_probs.append(token_lp.logprob if token_lp is not None else 0.0)

        response_text = hf_tokenizer.decode(action_ids, skip_special_tokens=False)

        test_setup, test_list = _parse_mbpp_label(label)

        target_func = None
        for t in test_list:
            m = re.search(r"(?:assert\s+)?(\w+)\s*\(", str(t))
            if m:
                target_func = m.group(1)
                break

        code = extract_code(response_text, is_truncated=is_truncated, target_func=target_func)

        if code:
            reward, pass_rate = compute_mbpp_reward(code, test_setup, test_list, is_truncated)
        else:
            reward, pass_rate = 0.0, 0.0

        return {
            "prompt": prompt, "label": label, "images": images, "mm_train_inputs": None,
            "observation_tokens": obs_ids, "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs, "truncated": is_truncated,
            "reward": reward, "scores": reward,
            "extra_logs": {
                "code_len": len(code) if code else 0,
                "code_extracted": 1 if code else 0,
                "pass_rate": pass_rate,
                "test_pass_rate": pass_rate,
            },
        }
