#!/usr/bin/env python3
# KernelBench PyTorch -> standalone CUDA C++ GRPO/QLoRA training
#
# DEBUG VERSION:
# - Uses only 3 training tasks and 2 eval tasks to verify the pipeline runs
# - Keeps the same overall structure:
#     PyTorch KernelBench task -> generate standalone CUDA C++ -> compile -> run -> compare
#
# Expected dataset layout:
#   <KERNELBENCH_ROOT>/
#       level1/*.py
#       level2/*.py
#       level3/*.py
#       level4/*.py
#
# Expected task API:
#   class Model(nn.Module)
#   get_init_inputs()
#   get_inputs()
#
# Example:
#   export CUDA_VISIBLE_DEVICES=0,1
#   export HUGGINGFACE_TOKEN=...
#   python train_kernelbench_qwen3_32b_cuda_grpo_debug.py
#
# Install:
#   pip install -U "transformers>=4.51.0" peft accelerate bitsandbytes huggingface_hub torch numpy

import os
import re
import math
import json
import shlex
import random
import tempfile
import subprocess
import statistics
import time
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from huggingface_hub import login
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

try:
    import bitsandbytes as bnb
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False


# =========================
# 0) CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen3-32B"
USE_4BIT_QLORA = True
TRAIN_DTYPE = torch.bfloat16

SYSTEM_PROMPT = "You are an expert CUDA C++ engineer."

SAVE_DIR = "checkpoints/kernelbench-qwen3-32b-cuda-qlora-grpo-debug"
SAVE_TOKENIZER = True
EXPORT_DIR = "kernelbench_generated_cuda_qwen3_32b_debug"
EVAL_JSONL_PATH = "kernelbench_eval_records_qwen3_32b_cuda_debug.jsonl"

KERNELBENCH_ROOT = "KernelBench"
DEFAULT_HARDWARE_TEXT = "NVIDIA A100 40GB CUDA GPU"

STRUCT_RANKER_PATH = "struct_ranker-64-min-speedup-1-epoch-150.json"

# Debug subset mode
DEBUG_SUBSET_MODE = True
N_TRAIN_TASKS_DEBUG = 30
N_EVAL_TASKS = 2
EVAL_SPLIT_SEED = 1337

# Faster smoke-test settings
N_INIT_VARIANTS = 1
N_REFINE_STEPS = 1
TEMPS = [0.2]
TOP_P = 0.9
MAX_STEP_TOK = 1536
MAX_LOGPROB_TOK = 3072

# GRPO
LAMBDA_GRPO = 1.0
CLIP_EPS = 0.2

# Reward
ALPHA_SPEED = 1.0
CLIP_LOG2_S = (-0.5, 4.0)
PENAL_FAIL_COMPILE = 3.0
PENAL_FAIL_WRONG = 2.0
PENAL_CV = 0.5
BONUS_KEEP_GOOD = 0.25
STRUCT_REWARD_WEIGHT = 0.2

# Optim
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10
TOTAL_STEPS = 100

# Timing / eval
TRIALS_PER_INPUT = 1
WARMUP_RUNS = 0
RUN_TIMEOUT_SEC = 240.0
N_CORRECTNESS_INPUTS = 1

# Tolerance
ATOL = 1e-4
RTOL = 1e-4

# Few-shot disabled
USE_FEWSHOT_CONTEXT = False
FEWSHOT_K = 2
FEWSHOT_MAX_CHARS_PER_CODE = 8000


# =========================
# 1) HELPERS
# =========================
def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else float("inf")


def _mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else float("inf")


def _stdev(xs: List[float]) -> float:
    if not xs or len(xs) <= 1:
        return 0.0
    return float(statistics.pstdev(xs))


def _cv(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = statistics.mean(xs)
    if m == 0:
        return 0.0
    return float(statistics.pstdev(xs) / m)


def seed_everything(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module_from_path(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n// ... [TRUNCATED]\n"


def _strip_md_fences(raw: str) -> str:
    raw = raw.replace("```cpp", "")
    raw = raw.replace("```c++", "")
    raw = raw.replace("```cu", "")
    raw = raw.replace("```cuda", "")
    raw = raw.replace("```", "")
    return raw.strip()


def _find_cuda_start(raw: str) -> str:
    markers = [
        "#include",
        "__global__",
        "int main",
        "using namespace std",
    ]
    start_idx = len(raw)
    for marker in markers:
        pos = raw.find(marker)
        if pos != -1 and pos < start_idx:
            start_idx = pos
    text = raw[start_idx:] if start_idx < len(raw) else raw
    last_brace = text.rfind("}")
    if last_brace != -1:
        text = text[:last_brace + 1]
    return text.strip()


def _run_cmd(
    cmd: List[str],
    timeout: float,
    stdin_data: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        out, err = proc.communicate(input=stdin_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return -1, "", "TIMEOUT"
    return proc.returncode, out, err


def compute_speedups(ref_medians_ms: List[float], gen_medians_ms: List[float]) -> Dict:
    n = min(len(ref_medians_ms), len(gen_medians_ms))
    per_input = []
    for i in range(n):
        ref_t = max(_safe_float(ref_medians_ms[i], float("inf")), 1e-9)
        gen_t = max(_safe_float(gen_medians_ms[i], float("inf")), 1e-9)
        per_input.append(ref_t / gen_t)
    speedup_first = per_input[0] if per_input else 0.0
    speedup_mean = (sum(per_input) / len(per_input)) if per_input else 0.0
    speedup_std = _stdev(per_input)
    return {
        "speedup_per_input": per_input,
        "speedup_first": speedup_first,
        "speedup_mean": speedup_mean,
        "speedup_std": speedup_std,
    }


# =========================
# 2) STRUCTURAL FEATURES
# =========================
RANKER = None
W_f = None
MEAN_f = None
STD_f = None

if os.path.exists(STRUCT_RANKER_PATH):
    with open(STRUCT_RANKER_PATH, "r", encoding="utf-8") as f:
        RANKER = json.load(f)


def maybe_init_struct_reward_tensors():
    global W_f, MEAN_f, STD_f
    if RANKER is None:
        return
    if W_f is None:
        W_f = torch.tensor(RANKER["weights"], device="cuda:0", dtype=torch.float32)
        MEAN_f = torch.tensor(RANKER["norm_mean"], device="cuda:0", dtype=torch.float32)
        STD_f = torch.tensor(RANKER["norm_std"], device="cuda:0", dtype=torch.float32)


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//.*?$", "", src, flags=re.MULTILINE)
    return src


KERNEL_DEF_RX = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*\{", re.DOTALL)
ARRAY_SUB_RX = re.compile(r"\b([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]")
ASSIGN_RX = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);")


def _find_matching_brace(s: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_kernels(src: str) -> List[Tuple[str, str]]:
    out = []
    for m in KERNEL_DEF_RX.finditer(src):
        name = m.group(1)
        body_start = m.end() - 1
        body_end = _find_matching_brace(src, body_start)
        if body_end == -1:
            continue
        body = src[body_start:body_end + 1]
        out.append((name, body))
    return out


def _build_symtab(kernel_body: str) -> Dict[str, str]:
    sym = {}
    for m in ASSIGN_RX.finditer(kernel_body):
        sym[m.group(1)] = m.group(2).strip()
    return sym


def _expand_expr(expr: str, sym: Dict[str, str], max_steps: int = 6) -> str:
    out = expr
    for _ in range(max_steps):
        changed = False
        for var, rhs in sym.items():
            new = re.sub(rf"\b{re.escape(var)}\b", f"({rhs})", out)
            if new != out:
                out = new
                changed = True
        if not changed:
            break
    return out


def _stride_wrt_threadidx_x(expr: str) -> Optional[int]:
    e = re.sub(r"\s+", "", expr)
    if "threadIdx.x" not in e:
        return 0
    if re.search(r"blockIdx\.x\*blockDim\.x\+threadIdx\.x", e):
        return 1
    m = re.search(r"threadIdx\.x\*(\d+)", e)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\*threadIdx\.x", e)
    if m:
        return int(m.group(1))
    if re.search(r"threadIdx\.x[<>\|\&\^\%\/]", e) or re.search(r"[<>\|\&\^\%\/]threadIdx\.x", e):
        return None
    return 1


def _coal_score(stride: Optional[int]) -> float:
    if stride is None:
        return 0.0
    if stride == 1:
        return 1.0
    if stride == 0:
        return 0.2
    if stride == 2:
        return 0.4
    if stride == 4:
        return 0.1
    if stride > 4:
        return -0.5
    return 0.0


def _count_ops(body: str) -> int:
    b = re.sub(r"\+\+", "", body)
    b = re.sub(r"--", "", b)
    plus = len(re.findall(r"\+", b))
    minus = len(re.findall(r"-", b))
    mult = len(re.findall(r"\*", b))
    div = len(re.findall(r"/", b))
    popc = len(re.findall(r"__popc(ll)?\b|__popcll\b", b))
    return max(0, plus + minus + mult + div + 4 * popc)


def _parse_threads_per_block(src: str) -> Optional[int]:
    m = re.search(r"\bthreadsPerBlock\s*=\s*(\d+)\s*;", src)
    if m:
        return int(m.group(1))
    m = re.search(r"<<<\s*[^,]+,\s*(\d+)\s*(?:,|>>> )", src)
    if m:
        return int(m.group(1))
    return None


def _count_syncthreads(body: str) -> int:
    return len(re.findall(r"__syncthreads\s*\(", body))


def _count_atomics(body: str) -> int:
    return len(re.findall(r"\batomic(Add|Exch|CAS|Max|Min|And|Or|Xor|Inc|Dec)\b", body))


def _transfer_proxy(host_src: str) -> float:
    memcpy_calls = len(re.findall(r"\bcudaMemcpy\s*\(", host_src))
    in_loop_memcpy = len(re.findall(r"\bfor\s*\(.*?\)\s*\{[^}]*\bcudaMemcpy\s*\(", host_src, flags=re.DOTALL))
    score = 0.0
    score -= 0.05 * max(0, memcpy_calls - 2)
    score -= 0.6 * in_loop_memcpy
    if memcpy_calls <= 2 and in_loop_memcpy == 0:
        score += 0.2
    return score


def extract_struct_features(cuda_src: str) -> Dict[str, float]:
    zero = {
        "coal": 0.0, "ai": 0.0, "occ": 0.0, "div": 0.0,
        "xfer": 0.0, "atomics": 0.0, "syncthreads": 0.0,
        "kernels": 0.0, "tpb": 0.0, "gmem_access": 0.0, "ops": 0.0
    }
    if not cuda_src or not cuda_src.strip():
        return zero

    src = _strip_comments(cuda_src)
    kernels = _extract_kernels(src)
    tpb = _parse_threads_per_block(src)
    xfer = _transfer_proxy(src)

    coal_list = []
    ai_list = []
    occ_list = []
    div_list = []

    atomics_total = 0
    sync_total = 0
    gmem_total = 0
    ops_total = 0

    for _, body in kernels:
        sym = _build_symtab(body)

        mem_scores = []
        gmem = 0

        for m in ARRAY_SUB_RX.finditer(body):
            idx_expr = _expand_expr(m.group(2), sym)
            stride = _stride_wrt_threadidx_x(idx_expr)
            mem_scores.append(_coal_score(stride))
            gmem += 1

        coal_k = (sum(mem_scores) / len(mem_scores)) if mem_scores else 0.0
        ops = _count_ops(body)

        ai_k = ops / float(gmem + 1e-9)
        ai_k = float(max(-1.0, min(1.0, math.tanh(0.25 * ai_k))))

        if tpb is None:
            occ_k = 0.0
        else:
            if 128 <= tpb <= 256:
                occ_k = 1.0
            elif 64 <= tpb < 128 or 256 < tpb <= 512:
                occ_k = 0.4
            else:
                occ_k = -0.3

        ifs = re.findall(r"\bif\s*\((.*?)\)", body, flags=re.DOTALL)
        div_k = 0.0
        for cond in ifs:
            c = re.sub(r"\s+", "", cond)
            if "idx<" in c or "idx>=" in c:
                continue
            if "threadIdx" in c or "blockIdx" in c:
                continue
            div_k -= 0.1

        atom = _count_atomics(body)
        syn = _count_syncthreads(body)

        atomics_total += atom
        sync_total += syn
        gmem_total += gmem
        ops_total += ops

        coal_list.append(coal_k)
        ai_list.append(ai_k)
        occ_list.append(occ_k)
        div_list.append(div_k)

    def avg(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "coal": avg(coal_list),
        "ai": avg(ai_list),
        "occ": avg(occ_list),
        "div": avg(div_list),
        "xfer": float(xfer),
        "atomics": float(atomics_total),
        "syncthreads": float(sync_total),
        "kernels": float(len(kernels)),
        "tpb": float(tpb if tpb is not None else 0.0),
        "gmem_access": float(gmem_total),
        "ops": float(ops_total),
    }


def structure_reward(features: Dict[str, float]) -> float:
    if RANKER is None:
        return 0.0
    maybe_init_struct_reward_tensors()
    x = torch.tensor(
        [float(features.get(k, 0.0)) for k in RANKER["feature_names"]],
        device="cuda:0",
        dtype=torch.float32,
    )
    x = (x - MEAN_f) / torch.clamp(STD_f, min=1e-8)
    return float((x @ W_f).item())


# =========================
# 3) DATA
# =========================
@dataclass
class KBTask:
    task_id: str
    level: int
    torch_code: str
    reference_path: str
    entry_point: str
    hardware_text: str
    extra: Dict = None


@dataclass
class FewshotExample:
    task_id: str
    torch_code: str
    cuda_code: str
    hardware_text: str
    score: float
    meta: Dict


def load_kernelbench_tasks(root: str, hardware_text: str = DEFAULT_HARDWARE_TEXT) -> List[KBTask]:
    tasks = []
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"KERNELBENCH_ROOT not found: {root}")

    level_dirs = [("level1", 1), ("level2", 2), ("level3", 3), ("level4", 4)]
    for level_name, level_num in level_dirs:
        level_dir = root_p / level_name
        if not level_dir.exists():
            continue
        for py_file in sorted(level_dir.glob("*.py")):
            if py_file.name.startswith("_") or py_file.name == "__init__.py":
                continue
            tasks.append(
                KBTask(
                    task_id=f"{level_name}_{py_file.stem}",
                    level=level_num,
                    torch_code=py_file.read_text(encoding="utf-8"),
                    reference_path=str(py_file),
                    entry_point="Model",
                    hardware_text=hardware_text,
                    extra={"level_dir": level_name, "file_name": py_file.name},
                )
            )

    if not tasks:
        raise RuntimeError("No KernelBench tasks found.")
    return tasks


def split_train_eval_tasks(tasks: List[KBTask], n_eval_tasks: int, seed: int = 1337):
    if len(tasks) <= n_eval_tasks:
        raise RuntimeError(
            f"Need more than {n_eval_tasks} tasks for a held-out eval set; found {len(tasks)}."
        )
    rng = random.Random(seed)
    idxs = list(range(len(tasks)))
    rng.shuffle(idxs)
    eval_idxs = set(idxs[:n_eval_tasks])
    train_tasks = [t for i, t in enumerate(tasks) if i not in eval_idxs]
    eval_tasks = [t for i, t in enumerate(tasks) if i in eval_idxs]
    return train_tasks, eval_tasks


def select_debug_subset(
    tasks: List[KBTask],
    n_train_tasks: int,
    n_eval_tasks: int,
    seed: int = 1337,
):
    total_needed = n_train_tasks + n_eval_tasks
    if len(tasks) < total_needed:
        raise RuntimeError(
            f"Need at least {total_needed} tasks for debug subset, but found only {len(tasks)}."
        )

    rng = random.Random(seed)
    idxs = list(range(len(tasks)))
    rng.shuffle(idxs)

    chosen = idxs[:total_needed]
    eval_idxs = set(chosen[:n_eval_tasks])
    train_idxs = set(chosen[n_eval_tasks:n_eval_tasks + n_train_tasks])

    train_tasks = [t for i, t in enumerate(tasks) if i in train_idxs]
    eval_tasks = [t for i, t in enumerate(tasks) if i in eval_idxs]
    return train_tasks, eval_tasks


def build_fewshot_context(pool: List[FewshotExample], current_task_id: str, k: int) -> str:
    if not USE_FEWSHOT_CONTEXT or k <= 0 or not pool:
        return ""
    cand = [ex for ex in pool if ex.task_id != current_task_id]
    if not cand:
        return ""
    cand.sort(key=lambda ex: ex.score, reverse=True)
    chosen = cand[:k]

    blocks = ["[FEWSHOT]"]
    for ex in chosen:
        blocks.append("[EXAMPLE]")
        blocks.append(f"// score≈{ex.score:.3f} task_id={ex.task_id}")
        blocks.append("[PYTORCH_CODE]")
        blocks.append(_truncate(ex.torch_code, FEWSHOT_MAX_CHARS_PER_CODE))
        blocks.append("[/PYTORCH_CODE]")
        blocks.append("[CUDA]")
        blocks.append(_truncate(ex.cuda_code, FEWSHOT_MAX_CHARS_PER_CODE))
        blocks.append("[/CUDA]")
        blocks.append("[HARDWARE]")
        blocks.append(_truncate(ex.hardware_text, 2000))
        blocks.append("[/HARDWARE]")
        blocks.append("[/EXAMPLE]\n")
    blocks.append("[/FEWSHOT]\n")
    return "\n".join(blocks)


def task_to_prompt(task: KBTask, fewshot_pool: Optional[List[FewshotExample]] = None) -> str:
    fewshot = ""
    if fewshot_pool:
        fewshot = build_fewshot_context(fewshot_pool, current_task_id=task.task_id, k=FEWSHOT_K)

    io_contract = r"""
Generate a standalone CUDA C++ program that can be compiled by nvcc as a single translation unit.

Output requirements:
- Output ONLY CUDA C++ code.
- Do NOT include markdown fences.
- Do NOT include explanations.
- The file must contain a main() function.

I/O contract for the generated program:
- Read from stdin using plain text.
- First token: N_INIT
- Then N_INIT arguments, one per line, in one of these formats:
    SCALAR_INT <value>
    SCALAR_FLOAT <value>
    TENSOR_FLOAT32 <ndim> <d1> <d2> ... <dk> <numel> <v1> <v2> ... <vN>
- Then token: N_INPUTS
- Then N_INPUTS arguments, one per line, in the same formats.

Output contract:
- Print:
    N_OUTPUTS <m>
- Then for each output tensor:
    TENSOR_FLOAT32 <ndim> <d1> <d2> ... <dk> <numel> <v1> <v2> ... <vN>

Semantic requirements:
- The program must implement the same computation as the PyTorch Model forward pass.
- Use CUDA kernels for the core computation.
- Use float32 for tensor values.
- Preserve correctness.
"""

    prompt = (
        fewshot
        + "Given the following KernelBench PyTorch reference program, generate an equivalent standalone CUDA C++ program.\n"
        + io_contract
        + "\n[PYTORCH_CODE]\n"
        + task.torch_code
        + "\n[/PYTORCH_CODE]\n"
        + f"[HARDWARE]\n{task.hardware_text}\n[/HARDWARE]\n"
    )
    return prompt


# =========================
# 4) POLICY
# =========================
SPECIALS = {
    "additional_special_tokens": [
        "[PYTORCH_CODE]", "[/PYTORCH_CODE]",
        "[HARDWARE]", "[/HARDWARE]",
        "[PREV_CUDA]", "[/PREV_CUDA]",
        "[FEWSHOT]", "[/FEWSHOT]",
        "[EXAMPLE]", "[/EXAMPLE]",
        "[CUDA]", "[/CUDA]",
    ]
}


class Policy(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

    def _build_chat_prompt(self, user_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n"

    @torch.no_grad()
    def generate_candidate(self, user_prompt: str, temperature: float, top_p: float) -> str:
        chat_prompt = self._build_chat_prompt(user_prompt)
        tok = self.tokenizer(chat_prompt, return_tensors="pt")
        tok.pop("token_type_ids", None)
        tok = {k: v.to("cuda:0") for k, v in tok.items()}
        input_len = tok["input_ids"].shape[1]

        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=MAX_STEP_TOK,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        out = self.model.generate(**tok, generation_config=gen_cfg)
        gen_ids = out[0][input_len:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        raw = _strip_md_fences(raw)
        raw = _find_cuda_start(raw)
        return raw

    def logprob_of_completion(self, user_prompt: str, completion_text: str) -> Tuple[torch.Tensor, Dict]:
        prompt_text = self._build_chat_prompt(user_prompt)
        full_text = prompt_text + completion_text

        tok = self.tokenizer(full_text, return_tensors="pt")
        tok.pop("token_type_ids", None)
        prompt_tok = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_tok.pop("token_type_ids", None)

        prompt_len = prompt_tok["input_ids"].shape[1]
        input_ids = tok["input_ids"][0]

        if input_ids.shape[0] > MAX_LOGPROB_TOK:
            cut = input_ids.shape[0] - MAX_LOGPROB_TOK
            for k in tok:
                tok[k] = tok[k][:, cut:]
            prompt_len = max(0, prompt_len - cut)

        tok = {k: v.to("cuda:0") for k, v in tok.items()}
        logits = self.model(**tok).logits[0]
        input_ids = tok["input_ids"][0]

        T = input_ids.shape[0]
        mask = torch.zeros(T, dtype=torch.bool, device=logits.device)
        if prompt_len < T:
            mask[prompt_len:] = True

        logits = logits[:-1]
        targets = input_ids[1:]
        mask = mask[1:]

        logp = F.log_softmax(logits.float(), dim=-1)
        sel = logp[mask, targets[mask]]
        total_logprob = sel.sum()
        return total_logprob, {"mask_count": int(mask.sum().item()), "len": int(T)}


# =========================
# 5) SERIALIZATION
# =========================
def _obj_to_spec_line(x: Any) -> str:
    if torch.is_tensor(x):
        t = x.detach().float().cpu().contiguous()
        shape = list(t.shape)
        vals = t.view(-1).tolist()
        shape_str = " ".join(str(int(d)) for d in shape)
        vals_str = " ".join("{:.9g}".format(float(v)) for v in vals)
        return f"TENSOR_FLOAT32 {len(shape)} {shape_str} {len(vals)} {vals_str}"
    if isinstance(x, bool):
        return f"SCALAR_INT {1 if x else 0}"
    if isinstance(x, int):
        return f"SCALAR_INT {x}"
    if isinstance(x, float):
        return f"SCALAR_FLOAT {x:.9g}"
    raise TypeError(f"Unsupported input type for serialization: {type(x)}")


def build_stdin_payload(init_inputs: Any, fwd_inputs: Any) -> str:
    if isinstance(init_inputs, tuple):
        init_list = list(init_inputs)
    elif isinstance(init_inputs, list):
        init_list = init_inputs
    elif init_inputs is None:
        init_list = []
    else:
        init_list = [init_inputs]

    if isinstance(fwd_inputs, tuple):
        fwd_list = list(fwd_inputs)
    elif isinstance(fwd_inputs, list):
        fwd_list = fwd_inputs
    elif fwd_inputs is None:
        fwd_list = []
    else:
        fwd_list = [fwd_inputs]

    lines = [f"N_INIT {len(init_list)}"]
    for x in init_list:
        lines.append(_obj_to_spec_line(x))
    lines.append(f"N_INPUTS {len(fwd_list)}")
    for x in fwd_list:
        lines.append(_obj_to_spec_line(x))
    lines.append("")
    return "\n".join(lines)


def flatten_ref_outputs(x: Any) -> List[torch.Tensor]:
    if torch.is_tensor(x):
        return [x.detach().float().cpu().contiguous()]
    if isinstance(x, (list, tuple)):
        out = []
        for y in x:
            out.extend(flatten_ref_outputs(y))
        return out
    if isinstance(x, dict):
        out = []
        for k in sorted(x.keys()):
            out.extend(flatten_ref_outputs(x[k]))
        return out
    raise TypeError(f"Unsupported output type from reference model: {type(x)}")


def parse_cuda_stdout(stdout_text: str) -> List[torch.Tensor]:
    toks = stdout_text.strip().split()
    if not toks:
        raise RuntimeError("Empty stdout from generated CUDA executable.")
    if toks[0] != "N_OUTPUTS":
        raise RuntimeError("Expected stdout to start with N_OUTPUTS.")
    idx = 1
    n_out = int(toks[idx]); idx += 1
    outs = []
    for _ in range(n_out):
        kind = toks[idx]; idx += 1
        if kind != "TENSOR_FLOAT32":
            raise RuntimeError(f"Unsupported output kind: {kind}")
        ndim = int(toks[idx]); idx += 1
        shape = []
        for _ in range(ndim):
            shape.append(int(toks[idx])); idx += 1
        numel = int(toks[idx]); idx += 1
        vals = [float(toks[idx + i]) for i in range(numel)]
        idx += numel
        t = torch.tensor(vals, dtype=torch.float32).reshape(shape)
        outs.append(t)
    return outs


# =========================
# 6) HARNESS
# =========================
@dataclass
class CandMetrics:
    compile_ok: bool
    pass_rate: float
    ref_medians_ms: List[float]
    gen_medians_ms: List[float]
    ref_stdev_ms: List[float]
    gen_stdev_ms: List[float]
    cv: float
    valid: bool
    correctness_trials: int
    features: Optional[Dict[str, float]] = None
    seq_outputs: Optional[List[str]] = None
    cuda_outputs: Optional[List[str]] = None
    compile_errs: Optional[List[str]] = None


class KernelBenchHarness:
    def __init__(
        self,
        rank_gpu: Optional[int] = 0,
        nvcc: str = "nvcc",
        nvccflags: str = "-O3",
        warmup_runs: int = WARMUP_RUNS,
        trials_per_input: int = TRIALS_PER_INPUT,
        correctness_inputs: int = N_CORRECTNESS_INPUTS,
        run_timeout_sec: float = RUN_TIMEOUT_SEC,
        atol: float = ATOL,
        rtol: float = RTOL,
    ):
        self.rank_gpu = rank_gpu
        self.nvcc = nvcc
        self.nvccflags = shlex.split(nvccflags)
        self.warmup_runs = warmup_runs
        self.trials_per_input = trials_per_input
        self.correctness_inputs = correctness_inputs
        self.run_timeout_sec = run_timeout_sec
        self.atol = atol
        self.rtol = rtol

    def _env(self) -> Dict[str, str]:
        if self.rank_gpu is None:
            return {}
        return {"CUDA_VISIBLE_DEVICES": str(self.rank_gpu)}

    def _compile_cuda(self, cu_path: str, out_path: str) -> Tuple[bool, str]:
        cmd = [self.nvcc, cu_path, "-o", out_path, *self.nvccflags]
        rc, out, err = _run_cmd(cmd, timeout=self.run_timeout_sec, extra_env=self._env())
        return (rc == 0, err if rc != 0 else "")

    def _instantiate_model(self, mod, class_name: str, init_inputs):
        cls = getattr(mod, class_name)
        if isinstance(init_inputs, (list, tuple)):
            obj = cls(*init_inputs)
        elif isinstance(init_inputs, dict):
            obj = cls(**init_inputs)
        elif init_inputs is None:
            obj = cls()
        else:
            obj = cls(init_inputs)
        obj = obj.cuda()
        obj.eval()
        return obj

    def _make_inputs(self, ref_mod):
        if hasattr(ref_mod, "get_init_inputs") and hasattr(ref_mod, "get_inputs"):
            init_inputs = ref_mod.get_init_inputs()
            sample_inputs = [ref_mod.get_inputs() for _ in range(self.correctness_inputs)]
            return init_inputs, sample_inputs
        raise RuntimeError("Expected get_init_inputs() and get_inputs().")

    def _time_reference_model(self, model, fwd_inputs) -> Tuple[float, float]:
        times = []
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                if isinstance(fwd_inputs, (list, tuple)):
                    _ = model(*[x.cuda() if torch.is_tensor(x) else x for x in fwd_inputs])
                elif isinstance(fwd_inputs, dict):
                    _ = model(**{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in fwd_inputs.items()})
                else:
                    _ = model(fwd_inputs.cuda() if torch.is_tensor(fwd_inputs) else fwd_inputs)
            torch.cuda.synchronize()

            for _ in range(self.trials_per_input):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                if isinstance(fwd_inputs, (list, tuple)):
                    _ = model(*[x.cuda() if torch.is_tensor(x) else x for x in fwd_inputs])
                elif isinstance(fwd_inputs, dict):
                    _ = model(**{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in fwd_inputs.items()})
                else:
                    _ = model(fwd_inputs.cuda() if torch.is_tensor(fwd_inputs) else fwd_inputs)
                end.record()
                torch.cuda.synchronize()
                times.append(float(start.elapsed_time(end)))

        return _median(times), _stdev(times)

    def _run_exe_once(self, exe: str, stdin_payload: str) -> Tuple[int, str, str, float]:
        start = time.perf_counter()
        proc = subprocess.Popen(
            [exe],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._env(),
        )
        try:
            out, err = proc.communicate(stdin_payload, timeout=self.run_timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            return -1, "", "TIMEOUT", (time.perf_counter() - start) * 1000.0
        t_ms = (time.perf_counter() - start) * 1000.0
        return proc.returncode, out, err, t_ms

    def _time_generated_exe(self, exe: str, stdin_payload: str) -> Tuple[float, float, str, str]:
        times = []
        final_out = ""
        final_err = ""

        for _ in range(self.warmup_runs):
            rc, out, err, _ = self._run_exe_once(exe, stdin_payload)
            if rc != 0:
                raise RuntimeError(f"Generated exe warmup failed: {err}")

        for _ in range(self.trials_per_input):
            rc, out, err, t_ms = self._run_exe_once(exe, stdin_payload)
            if rc != 0:
                raise RuntimeError(f"Generated exe run failed: {err}")
            times.append(t_ms)
            final_out = out
            final_err = err

        return _median(times), _stdev(times), final_out, final_err

    def _allclose_outputs(self, ref_outs: List[torch.Tensor], cand_outs: List[torch.Tensor]) -> bool:
        if len(ref_outs) != len(cand_outs):
            return False
        for a, b in zip(ref_outs, cand_outs):
            if a.shape != b.shape:
                return False
            if not torch.allclose(a.float(), b.float(), atol=self.atol, rtol=self.rtol):
                return False
        return True

    def _format_output_summary(self, outs: List[torch.Tensor]) -> str:
        return str([{"shape": tuple(t.shape), "dtype": str(t.dtype)} for t in outs])

    def run(self, cuda_code: str, task: KBTask) -> CandMetrics:
        compile_errs = []
        ref_medians = []
        gen_medians = []
        ref_stdevs = []
        gen_stdevs = []
        seq_outputs = []
        cuda_outputs = []
        features = extract_struct_features(cuda_code)

        try:
            ref_mod = load_module_from_path(task.reference_path, f"kb_ref_{task.task_id}")
        except Exception as e:
            return CandMetrics(
                compile_ok=False,
                pass_rate=0.0,
                ref_medians_ms=[],
                gen_medians_ms=[],
                ref_stdev_ms=[],
                gen_stdev_ms=[],
                cv=0.0,
                valid=False,
                correctness_trials=0,
                features=features,
                compile_errs=[f"Reference import failed: {repr(e)}"],
            )

        try:
            with tempfile.TemporaryDirectory() as td:
                cu_path = os.path.join(td, "kern.cu")
                exe_path = os.path.join(td, "kern.out")
                with open(cu_path, "w", encoding="utf-8") as f:
                    f.write(cuda_code)

                ok, err = self._compile_cuda(cu_path, exe_path)
                if not ok:
                    compile_errs.append("CUDA COMPILE ERROR:\n" + err[:4000])
                    return CandMetrics(
                        compile_ok=False,
                        pass_rate=0.0,
                        ref_medians_ms=[],
                        gen_medians_ms=[],
                        ref_stdev_ms=[],
                        gen_stdev_ms=[],
                        cv=0.0,
                        valid=False,
                        correctness_trials=0,
                        features=features,
                        compile_errs=compile_errs,
                    )

                init_inputs, sample_inputs = self._make_inputs(ref_mod)
                model = self._instantiate_model(ref_mod, "Model", init_inputs)

                passes = 0
                cvs = []

                for i, fwd_inputs in enumerate(sample_inputs):
                    with torch.no_grad():
                        if isinstance(fwd_inputs, (list, tuple)):
                            ref_out = model(*[x.cuda() if torch.is_tensor(x) else x for x in fwd_inputs])
                        elif isinstance(fwd_inputs, dict):
                            ref_out = model(**{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in fwd_inputs.items()})
                        else:
                            ref_out = model(fwd_inputs.cuda() if torch.is_tensor(fwd_inputs) else fwd_inputs)

                    ref_outs = flatten_ref_outputs(ref_out)
                    seq_outputs.append(self._format_output_summary(ref_outs))

                    stdin_payload = build_stdin_payload(init_inputs, fwd_inputs)

                    try:
                        gen_med, gen_std, cand_stdout, cand_stderr = self._time_generated_exe(exe_path, stdin_payload)
                        cand_outs = parse_cuda_stdout(cand_stdout)
                    except Exception as e:
                        cand_outs = []
                        gen_med, gen_std = float("inf"), 0.0

                    cuda_outputs.append(self._format_output_summary(cand_outs) if cand_outs else "PARSE_OR_RUN_FAIL")
                    ok_match = self._allclose_outputs(ref_outs, cand_outs)
                    if ok_match:
                        passes += 1

                    try:
                        ref_med, ref_std = self._time_reference_model(model, fwd_inputs)
                    except Exception:
                        ref_med, ref_std = float("inf"), 0.0

                    ref_medians.append(ref_med)
                    gen_medians.append(gen_med)
                    ref_stdevs.append(ref_std)
                    gen_stdevs.append(gen_std)

                    cv_ref = ref_std / max(ref_med, 1e-9) if math.isfinite(ref_med) else 0.0
                    cv_gen = gen_std / max(gen_med, 1e-9) if math.isfinite(gen_med) else 0.0
                    cvs.append(max(cv_ref, cv_gen))

                    print(
                        f"[HARNESS] task={task.task_id} sample={i} "
                        f"correct={ok_match} ref_ms={ref_med:.4f}±{ref_std:.4f} "
                        f"gen_ms={gen_med:.4f}±{gen_std:.4f}"
                    )

                pass_rate = passes / max(1, len(sample_inputs))
                cv = max(cvs) if cvs else 0.0
                valid = bool(ref_medians and gen_medians and pass_rate > 0.0)

                return CandMetrics(
                    compile_ok=True,
                    pass_rate=pass_rate,
                    ref_medians_ms=ref_medians,
                    gen_medians_ms=gen_medians,
                    ref_stdev_ms=ref_stdevs,
                    gen_stdev_ms=gen_stdevs,
                    cv=cv,
                    valid=valid,
                    correctness_trials=len(sample_inputs),
                    features=features,
                    seq_outputs=seq_outputs,
                    cuda_outputs=cuda_outputs,
                    compile_errs=compile_errs,
                )

        except Exception as e:
            compile_errs.append(repr(e))
            return CandMetrics(
                compile_ok=False,
                pass_rate=0.0,
                ref_medians_ms=[],
                gen_medians_ms=[],
                ref_stdev_ms=[],
                gen_stdev_ms=[],
                cv=0.0,
                valid=False,
                correctness_trials=0,
                features=features,
                seq_outputs=[],
                cuda_outputs=[],
                compile_errs=compile_errs,
            )


# =========================
# 7) REWARD + GRPO
# =========================
def reward_from_metrics(m: CandMetrics) -> float:
    if not m.compile_ok:
        return -PENAL_FAIL_COMPILE

    correctness_penalty = -PENAL_FAIL_WRONG * (1.0 - m.pass_rate)
    if m.pass_rate <= 0.0:
        return float(correctness_penalty)

    r_struct = structure_reward(m.features) if m.features else 0.0

    if m.ref_medians_ms and m.gen_medians_ms:
        ref_t = m.ref_medians_ms[0]
        gen_t = m.gen_medians_ms[0]
        speedup = max(ref_t / max(gen_t, 1e-6), 1e-3)

        r_speed = ALPHA_SPEED * float(
            torch.clamp(torch.tensor(math.log2(speedup)), *CLIP_LOG2_S)
        )
        return float(
            correctness_penalty
            + r_speed
            + STRUCT_REWARD_WEIGHT * r_struct
            - PENAL_CV * m.cv
        )

    return float(correctness_penalty + BONUS_KEEP_GOOD * m.pass_rate + STRUCT_REWARD_WEIGHT * r_struct)


def grpo_loss_backward(
    policy: "Policy",
    user_prompts: List[str],
    completions: List[str],
    old_logps: torch.Tensor,
    rewards: torch.Tensor,
) -> float:
    total_loss_val = 0.0
    for i, (p, c) in enumerate(zip(user_prompts, completions)):
        new_lp, _ = policy.logprob_of_completion(p, c)
        ratio = torch.exp(new_lp - old_logps[i])
        adv = rewards[i]

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
        sample_loss = -torch.min(unclipped, clipped)
        sample_loss.backward()
        total_loss_val += float(sample_loss.detach().item())

    return total_loss_val / max(1, len(completions))


# =========================
# 8) SELF-REFINEMENT
# =========================
def build_feedback(task: KBTask, metrics: CandMetrics) -> str:
    lines = []

    if not metrics.compile_ok:
        lines.append("The previous CUDA program failed to compile or execute.")
        if metrics.compile_errs:
            lines.append("Compiler/runtime errors:")
            for err in metrics.compile_errs:
                snippet = err.strip()
                if len(snippet) > 1200:
                    snippet = snippet[:1200] + "\n... [truncated]"
                lines.append(snippet)
        return "\n".join(lines)

    if metrics.pass_rate < 1.0:
        lines.append(
            f"The previous CUDA program was only correct on {metrics.pass_rate*100:.1f}% of tested samples."
        )
    else:
        lines.append("The previous CUDA program was correct on all tested samples.")

    if metrics.ref_medians_ms and metrics.gen_medians_ms:
        ref_t = metrics.ref_medians_ms[0]
        gen_t = metrics.gen_medians_ms[0]
        speedup = ref_t / max(gen_t, 1e-6)
        ref_std = metrics.ref_stdev_ms[0] if metrics.ref_stdev_ms else 0.0
        gen_std = metrics.gen_stdev_ms[0] if metrics.gen_stdev_ms else 0.0
        lines.append(
            f"Reference time: {ref_t:.4f}±{ref_std:.4f} ms, "
            f"generated CUDA time: {gen_t:.4f}±{gen_std:.4f} ms "
            f"(speedup ≈ {speedup:.3f}x)."
        )

    if metrics.features:
        lines.append(f"Extracted structural features: {metrics.features}")

    if metrics.seq_outputs and metrics.cuda_outputs and metrics.pass_rate < 1.0:
        lines.append("Reference output summary:")
        lines.append(metrics.seq_outputs[0])
        lines.append("Generated CUDA output summary:")
        lines.append(metrics.cuda_outputs[0])

    lines.append("Generate a new corrected standalone CUDA C++ program only.")
    return "\n".join(lines)


def generate_population_and_refine(
    policy: "Policy",
    harness: "KernelBenchHarness",
    task: KBTask,
    fewshot_pool: Optional[List[FewshotExample]] = None,
    n_init: int = N_INIT_VARIANTS,
    n_refine: int = N_REFINE_STEPS,
) -> Tuple[List[str], List[CandMetrics], List[str]]:
    completions = []
    metrics_list = []
    prompts_used = []

    base_user_prompt = task_to_prompt(task, fewshot_pool=fewshot_pool)

    init_completions = []
    init_metrics = []

    for i in range(max(1, n_init)):
        temp = TEMPS[i % len(TEMPS)]
        cand = policy.generate_candidate(base_user_prompt, temperature=temp, top_p=TOP_P)
        m = harness.run(cand, task)

        init_completions.append(cand)
        init_metrics.append(m)
        prompts_used.append(base_user_prompt)

    completions.extend(init_completions)
    metrics_list.extend(init_metrics)

    rewards_init = [reward_from_metrics(m) for m in init_metrics]
    best_idx = max(range(len(rewards_init)), key=lambda idx: rewards_init[idx])

    prev_completion = init_completions[best_idx]
    prev_metric = init_metrics[best_idx]

    for r in range(max(0, n_refine)):
        fb = build_feedback(task, prev_metric)
        refined_user_prompt = (
            base_user_prompt
            + "\n\n[PREV_CUDA]\n"
            + prev_completion
            + "\n[/PREV_CUDA]\n\n"
            + fb
            + "\n"
        )

        temp = TEMPS[(n_init + r) % len(TEMPS)]
        cand = policy.generate_candidate(refined_user_prompt, temperature=temp, top_p=TOP_P)
        m = harness.run(cand, task)

        prompts_used.append(refined_user_prompt)
        completions.append(cand)
        metrics_list.append(m)

        prev_completion = cand
        prev_metric = m

        if m.compile_ok and m.pass_rate >= 1.0:
            break

    return completions, metrics_list, prompts_used


# =========================
# 9) TRAIN STEP
# =========================
def rlaif_step_single_process(
    policy: "Policy",
    harness: "KernelBenchHarness",
    task: KBTask,
    fewshot_pool: Optional[List[FewshotExample]] = None,
) -> Dict:
    completions, metrics, prompts_used = generate_population_and_refine(
        policy=policy,
        harness=harness,
        task=task,
        fewshot_pool=fewshot_pool,
        n_init=N_INIT_VARIANTS,
        n_refine=N_REFINE_STEPS,
    )

    with torch.no_grad():
        old_lps = torch.stack(
            [policy.logprob_of_completion(p, c)[0] for p, c in zip(prompts_used, completions)]
        ).to("cuda:0")

    rewards = torch.tensor(
        [reward_from_metrics(m) for m in metrics],
        device="cuda:0",
        dtype=torch.float32,
    )

    loss_grpo_val = grpo_loss_backward(
        policy=policy,
        user_prompts=prompts_used,
        completions=completions,
        old_logps=old_lps,
        rewards=rewards,
    )

    return {
        "loss_grpo": float(loss_grpo_val),
        "rewards_mean": float(rewards.mean().item()),
        "best_reward": float(rewards.max().item()),
    }


# =========================
# 10) MODEL BUILD
# =========================
def _select_lora_targets(model, wanted: List[str]) -> List[str]:
    present = set()
    for name, _ in model.named_modules():
        present.add(name.split(".")[-1])
    keep = [w for w in wanted if w in present]
    if not keep:
        for w in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if w in present:
                keep.append(w)
    return keep


def build_policy() -> Policy:
    tok = os.environ.get("HUGGINGFACE_TOKEN", "")
    if tok:
        login(token=tok)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens(SPECIALS)

    max_mem = {0: "38GiB", 1: "38GiB"} if torch.cuda.device_count() >= 2 else None

    if USE_4BIT_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=TRAIN_DTYPE,
            device_map="auto",
            max_memory=max_mem,
            quantization_config=bnb_config,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=TRAIN_DTYPE,
            device_map="auto",
            max_memory=max_mem,
        )
        model.resize_token_embeddings(len(tokenizer))

    wanted = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    targets = _select_lora_targets(model, wanted)
    print(f"[INFO] LoRA target_modules = {targets}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return Policy(model, tokenizer)


# =========================
# 11) EVAL RECORDS / EXPORT
# =========================
def append_eval_record_jsonl(
    jsonl_path: str,
    task: KBTask,
    best_completion: str,
    best_metrics: CandMetrics,
    eval_dataset_size: int,
):
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

    sp = compute_speedups(best_metrics.ref_medians_ms, best_metrics.gen_medians_ms)
    record = {
        "task_id": task.task_id,
        "level": task.level,
        "reference_path": task.reference_path,
        "pytorch_Version": torch.__version__,
        "inputs_tested": int(eval_dataset_size),
        "generated_cuda_version": "standalone_cuda_cpp",
        "achieved_speedup": _safe_float(sp["speedup_mean"]),
        "achieved_speedup_std": _safe_float(sp["speedup_std"]),
        "pytorch_median_ms": _safe_float(_mean(best_metrics.ref_medians_ms)),
        "pytorch_std_ms": _safe_float(_mean(best_metrics.ref_stdev_ms)),
        "generated_median_ms": _safe_float(_mean(best_metrics.gen_medians_ms)),
        "generated_std_ms": _safe_float(_mean(best_metrics.gen_stdev_ms)),
        "pass_rate": _safe_float(best_metrics.pass_rate),
        "compile_ok": bool(best_metrics.compile_ok),
    }

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def export_task_result_json(
    out_dir: str,
    task: KBTask,
    best_completion: str,
    best_metrics: CandMetrics,
    reward: float,
    extra: Optional[Dict] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    sp = compute_speedups(best_metrics.ref_medians_ms, best_metrics.gen_medians_ms)

    payload = {
        "task_id": task.task_id,
        "reference_path": task.reference_path,
        "reward": _safe_float(reward),
        "compile_ok": bool(best_metrics.compile_ok),
        "valid": bool(best_metrics.valid),
        "pass_rate": _safe_float(best_metrics.pass_rate),
        "cv": _safe_float(best_metrics.cv),
        "correctness_trials": int(best_metrics.correctness_trials),
        "ref_medians_ms": [_safe_float(x) for x in (best_metrics.ref_medians_ms or [])],
        "gen_medians_ms": [_safe_float(x) for x in (best_metrics.gen_medians_ms or [])],
        "ref_stdev_ms": [_safe_float(x) for x in (best_metrics.ref_stdev_ms or [])],
        "gen_stdev_ms": [_safe_float(x) for x in (best_metrics.gen_stdev_ms or [])],
        "features": best_metrics.features or {},
        **sp,
        "candidate_code": best_completion,
        "compile_errs": best_metrics.compile_errs or [],
    }
    if extra:
        payload["extra"] = extra

    out_json = os.path.join(out_dir, f"{task.task_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_cu = os.path.join(out_dir, f"{task.task_id}.cu")
    with open(out_cu, "w", encoding="utf-8") as f:
        f.write(best_completion)

    print(f"[EXPORT] Wrote {out_json}")
    print(f"[EXPORT] Wrote {out_cu}")


@torch.no_grad()
def final_evaluate(
    policy: "Policy",
    harness: "KernelBenchHarness",
    tasks: List[KBTask],
    fewshot_pool: Optional[List[FewshotExample]],
    eval_dataset_size: int,
) -> None:
    print("\n====================")
    print("Running final evaluation on held-out tasks...")
    print("====================\n")

    policy.eval()

    total_compile_ok = 0
    fully_correct_cases = 0
    faster_any_correct = 0
    faster_full_correct = 0
    total_speedup_any_correct = 0.0
    total_speedup_full_correct = 0.0

    with open(EVAL_JSONL_PATH, "w", encoding="utf-8") as _:
        pass

    eval_temp = 0.2
    eval_top_p = TOP_P

    for task in tasks:
        user_prompt = task_to_prompt(task, fewshot_pool=fewshot_pool)
        candidate = policy.generate_candidate(user_prompt, temperature=eval_temp, top_p=eval_top_p)
        metrics = harness.run(candidate, task)

        if metrics.compile_ok:
            total_compile_ok += 1

        if metrics.pass_rate >= 1.0:
            fully_correct_cases += 1

        if metrics.ref_medians_ms and metrics.gen_medians_ms and metrics.pass_rate > 0.0:
            sp = compute_speedups(metrics.ref_medians_ms, metrics.gen_medians_ms)
            speed = sp["speedup_mean"]
            total_speedup_any_correct += speed
            if speed > 1.0:
                faster_any_correct += 1

        if metrics.ref_medians_ms and metrics.gen_medians_ms and metrics.pass_rate >= 1.0:
            sp = compute_speedups(metrics.ref_medians_ms, metrics.gen_medians_ms)
            speed = sp["speedup_mean"]
            total_speedup_full_correct += speed
            if speed > 1.0:
                faster_full_correct += 1

        append_eval_record_jsonl(
            jsonl_path=EVAL_JSONL_PATH,
            task=task,
            best_completion=candidate,
            best_metrics=metrics,
            eval_dataset_size=eval_dataset_size,
        )

    avg_speedup_any_correct = total_speedup_any_correct / max(1, total_compile_ok) if total_compile_ok > 0 else 0.0
    avg_speedup_full_correct = total_speedup_full_correct / max(1, fully_correct_cases) if fully_correct_cases > 0 else 0.0

    print("\n==================== FINAL EVAL REPORT ====================")
    print(f"Model                                      : {MODEL_NAME}")
    print(f"Held-out eval dataset size                 : {eval_dataset_size}")
    print(f"Tasks compile/import OK                    : {total_compile_ok}")
    print(f"Tasks fully correct                        : {fully_correct_cases}")
    print(f"Tasks faster than reference (pass_rate>0)  : {faster_any_correct}")
    print(f"Tasks faster than reference (pass_rate=1)  : {faster_full_correct}")
    print(f"Average speedup over compiled tasks        : {avg_speedup_any_correct:.3f}")
    print(f"Average speedup over fully-correct tasks   : {avg_speedup_full_correct:.3f}")
    print(f"Evaluation JSONL                           : {EVAL_JSONL_PATH}")
    print("===========================================================\n")

    policy.train()


@torch.no_grad()
def optimize_and_export_all_tasks(
    policy: "Policy",
    harness: "KernelBenchHarness",
    tasks: List[KBTask],
    fewshot_pool: Optional[List[FewshotExample]],
    out_dir: str,
):
    print("\n====================")
    print("Optimizing + exporting per-task best CUDA programs...")
    print("====================\n")

    policy.eval()

    for task in tasks:
        completions, metrics_list, _prompts_used = generate_population_and_refine(
            policy=policy,
            harness=harness,
            task=task,
            fewshot_pool=fewshot_pool,
            n_init=N_INIT_VARIANTS,
            n_refine=N_REFINE_STEPS,
        )

        rewards = [reward_from_metrics(m) for m in metrics_list]
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])

        best_completion = completions[best_idx]
        best_metrics = metrics_list[best_idx]
        best_reward = rewards[best_idx]

        export_task_result_json(
            out_dir=out_dir,
            task=task,
            best_completion=best_completion,
            best_metrics=best_metrics,
            reward=best_reward,
            extra={"level": task.level},
        )

    policy.train()


def save_policy(policy: "Policy", save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    print(f"[SAVE] Saving LoRA adapters to {save_dir}")
    policy.model.save_pretrained(save_dir)
    if SAVE_TOKENIZER:
        print(f"[SAVE] Saving tokenizer to {save_dir}")
        policy.tokenizer.save_pretrained(save_dir)


# =========================
# 12) MAIN
# =========================
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA devices visible.")

    seed_everything(1337)

    if RANKER is None:
        print(f"[WARN] Structural ranker file not found at {STRUCT_RANKER_PATH}. Structural reward will be 0.")
    else:
        print(f"[INFO] Loaded structural ranker from {STRUCT_RANKER_PATH}")

    policy = build_policy()
    policy.train()

    if HAVE_BNB:
        optimizer = bnb.optim.PagedAdamW8bit(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, TOTAL_STEPS)

    harness = KernelBenchHarness(
        rank_gpu=0,
        warmup_runs=WARMUP_RUNS,
        trials_per_input=TRIALS_PER_INPUT,
        correctness_inputs=N_CORRECTNESS_INPUTS,
        run_timeout_sec=RUN_TIMEOUT_SEC,
        atol=ATOL,
        rtol=RTOL,
    )

    print(f"[INFO] Loading tasks from: {KERNELBENCH_ROOT}")
    all_tasks = load_kernelbench_tasks(KERNELBENCH_ROOT, hardware_text=DEFAULT_HARDWARE_TEXT)

    if DEBUG_SUBSET_MODE:
        train_tasks, eval_tasks = select_debug_subset(
            all_tasks,
            n_train_tasks=N_TRAIN_TASKS_DEBUG,
            n_eval_tasks=N_EVAL_TASKS,
            seed=EVAL_SPLIT_SEED,
        )
    else:
        train_tasks, eval_tasks = split_train_eval_tasks(
            all_tasks,
            n_eval_tasks=N_EVAL_TASKS,
            seed=EVAL_SPLIT_SEED,
        )

    fewshot_pool: List[FewshotExample] = []

    print(f"[INFO] Total KernelBench tasks = {len(all_tasks)}")
    print(f"[INFO] Train tasks            = {len(train_tasks)}")
    print(f"[INFO] Eval tasks             = {len(eval_tasks)}")

    print("[INFO] Training subset:")
    for t in train_tasks:
        print(f"  - {t.task_id}")

    print("[INFO] Eval subset:")
    for t in eval_tasks:
        print(f"  - {t.task_id}")

    num_epochs = 1
    print_every = 1
    step = 0

    for epoch in range(num_epochs):
        random.shuffle(train_tasks)
        for task in train_tasks:
            optimizer.zero_grad(set_to_none=True)

            out = rlaif_step_single_process(
                policy=policy,
                harness=harness,
                task=task,
                fewshot_pool=fewshot_pool,
            )

            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % print_every == 0:
                print(
                    f"[{step}] grpo_loss={out['loss_grpo']:.3f} | "
                    f"R_mean={out['rewards_mean']:.3f} bestR={out['best_reward']:.3f} | "
                    f"task={task.task_id}"
                )
            step += 1

    print("Training finished.")

    final_evaluate(
        policy=policy,
        harness=harness,
        tasks=eval_tasks,
        fewshot_pool=fewshot_pool,
        eval_dataset_size=len(eval_tasks),
    )

    optimize_and_export_all_tasks(
        policy=policy,
        harness=harness,
        tasks=eval_tasks,
        fewshot_pool=fewshot_pool,
        out_dir=EXPORT_DIR,
    )

    save_policy(policy, SAVE_DIR)
    print("[SAVE] Done.")


if __name__ == "__main__":
    main()
