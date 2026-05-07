#!/usr/bin/env python3
# Full run: BabelTower + GRPO (Qwen3-32B QLoRA + LoRA) on FULL dataset
#
# Install:
#   pip install -U "transformers>=4.45.0" peft accelerate bitsandbytes huggingface_hub torch numpy datasets
#
# Run (2× A100 40GB):
#   export CUDA_VISIBLE_DEVICES=0,1
#   export HUGGINGFACE_TOKEN=...
#   python train_qwen3_32b_qlora_grpo_babeltower_full.py \
#       --train_split train --eval_split test \
#       --total_steps 10000 --epochs 1 \
#       --export_dir optimized_cuda_outputs_babeltower_full \
#       --save_dir checkpoints/qwen3-32b-qlora-grpo-babeltower-full
#
# Notes:
# - Single-process training (NO DDP). Model is sharded across 2 GPUs via device_map="auto".
# - Harness pins compiled CUDA runs to GPU0 (CUDA_VISIBLE_DEVICES=0) to avoid interfering with training sharding.
# - Correctness is signature-line equality from BabelTower consistent_*_inputs snippets.
# - Speedup is wall-time of CUDA driver vs C++ driver (includes overhead). Good for sanity & pass@k.
# - Structure ranker shaping is optional; set --no_struct_ranker to disable.

import os
import math
import json
import random
import re
import shlex
import subprocess
import tempfile
import statistics
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset

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
# Defaults (FULL CONFIG)
# =========================
MODEL_NAME = "Qwen/Qwen3-32B"
BABEL_DATASET = "kcxain/BabelTower"

SYSTEM_PROMPT = "You are an expert CUDA C++ engineer."
USE_NO_THINK = True

# BabelTower
ALL_TASKS_LIMIT = 5

# Exploration + refinement (full)
N_INIT_VARIANTS = 4
N_REFINE_STEPS = 3
TEMPS = [0.2, 0.7]
TOP_P = 0.9
MAX_STEP_TOK = 1536
MAX_LOGPROB_TOK = 1024

# QLoRA
USE_4BIT_QLORA = True
TRAIN_DTYPE = torch.bfloat16

# Harness
TRIALS_PER_INPUT = 3
RUN_TIMEOUT_SEC = 240.0

# Reward
ALPHA_SPEED = 1.0
CLIP_LOG2_S = (-0.5, 4.0)
PENAL_FAIL_COMPILE = 3.0
PENAL_FAIL_WRONG = 2.0
PENAL_CV = 0.5
BONUS_KEEP_GOOD = 0.5
CLIP_EPS = 0.2

# Optim
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
TOTAL_STEPS = 10_000
EPOCHS = 1
PRINT_EVERY = 1

# Structure shaping
USE_STRUCT_RANKER = True
RANKER_PATH = "struct_ranker-64-min-speedup-1-epoch-150.json"
STRUCT_COEF = 0.2  # shaping weight

# Saving / export
SAVE_TOKENIZER = True


# =========================
# Helpers
# =========================
def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _median_ms(samples: List[float]) -> float:
    return float(statistics.median(samples)) if samples else float("inf")

def _cv(samples: List[float]) -> float:
    if not samples:
        return 0.0
    m = statistics.mean(samples)
    if m == 0:
        return 0.0
    return float(statistics.pstdev(samples) / m)

def compute_speedups(cpu_medians_ms: List[float], gpu_medians_ms: List[float]) -> Dict:
    n = min(len(cpu_medians_ms), len(gpu_medians_ms))
    per_input = []
    for i in range(n):
        cpu = max(_safe_float(cpu_medians_ms[i], float("inf")), 1e-9)
        gpu = max(_safe_float(gpu_medians_ms[i], float("inf")), 1e-9)
        per_input.append(cpu / gpu)
    return {
        "speedup_per_input": per_input,
        "speedup_first": per_input[0] if per_input else 0.0,
        "speedup_mean": (sum(per_input) / len(per_input)) if per_input else 0.0,
    }

def _infer_expected_wrapper_name(cuda_tests: List[str]) -> Optional[str]:
    rx = re.compile(r"\bwrapper\s*\(\s*([A-Za-z_]\w*)\s*,")
    for t in cuda_tests or []:
        m = rx.search(t)
        if m:
            return m.group(1)
    return None

def _infer_base_fn_name(cpp_code: str) -> str:
    # naive but works for BabelTower’s “single function” style
    m = re.search(r"\b[A-Za-z_]\w*\s+([A-Za-z_]\w*)\s*\(", cpp_code)
    return m.group(1) if m else "kernel_func"

def _delex_names(src: str, fn: str, wrapper: str) -> str:
    # Replace whole-word occurrences only
    src = re.sub(rf"\b{re.escape(fn)}\b", "KERNEL_FN", src)
    src = re.sub(rf"\b{re.escape(wrapper)}\b", "WRAPPER_FN", src)
    return src

# =========================
# Structure features + ranker
# =========================
RANKER: Optional[Dict[str, Any]] = None
W_f = MEAN_f = STD_f = None

def _load_ranker_if_any(path: str):
    global RANKER, W_f, MEAN_f, STD_f
    try:
        with open(path, "r", encoding="utf-8") as f:
            RANKER = json.load(f)
        W_f = torch.tensor(RANKER["weights"], device="cuda:0")
        MEAN_f = torch.tensor(RANKER["norm_mean"], device="cuda:0")
        STD_f = torch.tensor(RANKER["norm_std"], device="cuda:0")
        print(f"[INFO] Loaded struct ranker: {path} | d={len(RANKER.get('feature_names', []))}")
    except Exception as e:
        print(f"[WARN] Could not load ranker '{path}': {e}")
        RANKER = None

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

def _extract_kernels(src: str):
    out = []
    for m in KERNEL_DEF_RX.finditer(src):
        name = m.group(1)
        body_start = m.end() - 1
        body_end = _find_matching_brace(src, body_start)
        if body_end == -1:
            continue
        out.append((name, src[body_start:body_end + 1]))
    return out

def _build_symtab(kernel_body: str) -> Dict[str, str]:
    sym: Dict[str, str] = {}
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
    return max(
        0,
        len(re.findall(r"\+", b)) +
        len(re.findall(r"-", b)) +
        len(re.findall(r"\*", b)) +
        len(re.findall(r"/", b))
    )

def extract_struct_features(cuda_src: str) -> Dict[str, float]:
    if not cuda_src or not cuda_src.strip():
        return {"coal": 0.0, "ai": 0.0, "occ": 0.0, "div": 0.0, "xfer": 0.0,
                "atomics": 0.0, "syncthreads": 0.0, "kernels": 0.0, "tpb": 0.0, "gmem_access": 0.0, "ops": 0.0}

    src = _strip_comments(cuda_src)
    kernels = _extract_kernels(src)

    coal_list, ai_list, occ_list, div_list = [], [], [], []
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

        occ_k = 0.0
        div_k = -0.1 * len(re.findall(r"\bif\s*\(", body))

        coal_list.append(coal_k)
        ai_list.append(ai_k)
        occ_list.append(occ_k)
        div_list.append(div_k)
        gmem_total += gmem
        ops_total += ops

    def avg(xs): return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "coal": avg(coal_list),
        "ai": avg(ai_list),
        "occ": avg(occ_list),
        "div": avg(div_list),
        "xfer": 0.0,
        "atomics": 0.0,
        "syncthreads": 0.0,
        "kernels": float(len(kernels)),
        "tpb": 0.0,
        "gmem_access": float(gmem_total),
        "ops": float(ops_total),
    }

def structure_reward(features: Dict[str, float]) -> float:
    if not USE_STRUCT_RANKER or not RANKER or W_f is None:
        return 0.0
    x = torch.tensor([features.get(k, 0.0) for k in RANKER["feature_names"]], device="cuda:0")
    x = (x - MEAN_f) / (STD_f + 1e-12)
    return float((x @ W_f).item())


# =========================
# Data
# =========================
@dataclass
class Task:
    task_id: str
    cpp_code: str
    extra: Dict[str, Any]
    ref_cuda_full: str = ""   # cuda_code + "\n\n" + cuda_wrapper (few-shot solution)

def load_tasks_from_babeltower(split: str, limit: Optional[int] = None) -> List[Task]:
    ds = load_dataset(BABEL_DATASET, split=split)
    tasks: List[Task] = []
    for i, rec in enumerate(ds):
        if limit is not None and len(tasks) >= limit:
            break

        cpp = (rec.get("cpp_code") or "").strip()
        if not cpp:
            continue

        cuda_kernel = (rec.get("cuda_code") or "").strip()
        cuda_wrap = (rec.get("cuda_wrapper") or "").strip()

        ref_cuda_full = ""
        if cuda_kernel and cuda_wrap:
            ref_cuda_full = cuda_kernel + "\n\n" + cuda_wrap
        elif cuda_kernel:
            ref_cuda_full = cuda_kernel
        elif cuda_wrap:
            ref_cuda_full = cuda_wrap

        tasks.append(Task(
            task_id=str(rec.get("id", i)),
            cpp_code=cpp,
            extra={
                "consistent_cpp_inputs": rec.get("consistent_cpp_inputs") or [],
                "consistent_cuda_inputs": rec.get("consistent_cuda_inputs") or [],
                "consistent_outputs": rec.get("consistent_outputs") or [],
            },
            ref_cuda_full=ref_cuda_full,
        ))
    return tasks

class FewshotPool:
    def __init__(self, tasks: List[Task], seed: int = 123):
        self.rng = random.Random(seed)
        self.examples = [t for t in tasks if isinstance(t.ref_cuda_full, str) and t.ref_cuda_full.strip()]

    def sample(self, k: int, exclude_task_id: str) -> List[Task]:
        if k <= 0 or not self.examples:
            return []
        pool = [t for t in self.examples if t.task_id != exclude_task_id]
        if not pool:
            return []
        if len(pool) <= k:
            return pool
        return self.rng.sample(pool, k)

def _format_fewshot_example(ex_task: Task) -> str:
    fn_ex = _infer_base_fn_name(ex_task.cpp_code)
    cuda_tests_ex = ex_task.extra.get("consistent_cuda_inputs", [])
    expected_wrapper_ex = _infer_expected_wrapper_name(cuda_tests_ex) or f"{fn_ex}_cuda_invoke_in_cpp"

    return (
        "=== FEWSHOT EXAMPLE ===\n"
        f"// Function name: {fn_ex}\n"
        f"// Wrapper required by tests: {expected_wrapper_ex}\n"
        "[SEQ_CODE]\n"
        f"{ex_task.cpp_code}\n"
        "[/SEQ_CODE]\n"
        "[CUDA]\n"
        f"{ex_task.ref_cuda_full}\n"
        "[/CUDA]\n"
        "=== END FEWSHOT ===\n\n"
    )

def task_to_prompt(task: Task, fewshot_pool: Optional[FewshotPool] = None, fewshot_k: int = 0, fewshot_max_chars: int = 12000) -> str:
    
    fn = _infer_base_fn_name(task.cpp_code)

    cuda_tests = task.extra.get("consistent_cuda_inputs", [])
    expected_wrapper = _infer_expected_wrapper_name(cuda_tests) or f"{fn}_cuda_invoke_in_cpp"

    fewshot_block = ""
    if fewshot_pool is not None and fewshot_k > 0:
        exs = fewshot_pool.sample(fewshot_k, exclude_task_id=task.task_id)
        chunks = []
        total = 0
        for ex in exs:
            chunk = _format_fewshot_example(ex)
            if total + len(chunk) > fewshot_max_chars:
                break
            chunks.append(chunk)
            total += len(chunk)
        fewshot_block = "".join(chunks)

    user = (
        "You are given a C/C++ function (no main). Convert it to CUDA.\n"
        "REQUIREMENTS:\n"
        f"1) Keep the same function name: {fn}\n"
        f"2) Implement: __global__ void {fn}(...) with the SAME parameter list as the C function.\n"
        f"3) Implement a HOST wrapper EXACTLY named:\n"
        f"   void {expected_wrapper}(same parameters)\n"
        "   The wrapper must: cudaMalloc, cudaMemcpy H2D inputs, launch kernel, cudaMemcpy D2H outputs, cudaFree.\n"
        "4) Output ONLY CUDA C++ code (no markdown, no explanations).\n"
        "5) Do NOT write a main().\n"
        "6) Must compile with nvcc as a single translation unit.\n"
        "7) Your CUDA output MUST include BOTH the __global__ kernel AND the host wrapper function.\n\n"
    )

    if fewshot_block:
        user += (
            "Below are a few reference conversions from the dataset. "
            "Follow the same style and constraints, but adapt to the new function.\n\n"
            + fewshot_block
        )

    user += f"[SEQ_CODE]\n{task.cpp_code}\n[/SEQ_CODE]\n"

    if USE_NO_THINK:
        user += "\n/no_think\n"
    return user


# =========================
# Policy
# =========================
SPECIALS = {"additional_special_tokens": ["[SEQ_CODE]", "[/SEQ_CODE]", "[DECISION]", "[/DECISION]", "[CUDA]", "[/CUDA]", "[LAUNCH]", "[/LAUNCH]", "[PREV_CUDA]", "[/PREV_CUDA]"]}
TAG_RX = {
    "CUDA_CODE": re.compile(r"\[CUDA\](.*?)\[/CUDA\]", re.DOTALL),
    "DECISION": re.compile(r"\[DECISION\](.*?)\[/DECISION\]", re.DOTALL),
}

def _extract_first(text: str, rx: re.Pattern) -> str:
    m = rx.search(text)
    return (m.group(1).strip() if m else "").strip()

class Policy(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tid = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIALS["additional_special_tokens"]}
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

    def _build_chat_prompt(self, user_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n"

    @torch.no_grad()
    def generate_candidate(self, user_prompt: str, temperature: float, top_p: float) -> str:
        chat = self._build_chat_prompt(user_prompt)
        tok = self.tokenizer(chat, return_tensors="pt")
        tok.pop("token_type_ids", None)
        tok = {k: v.to("cuda:0") for k, v in tok.items()}
        in_len = tok["input_ids"].shape[1]

        cfg = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=MAX_STEP_TOK,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out = self.model.generate(**tok, generation_config=cfg)
        gen = out[0][in_len:]
        raw = self.tokenizer.decode(gen, skip_special_tokens=False)
        raw = raw.replace("```cpp", "").replace("```c++", "").replace("```cu", "").replace("```cuda", "").replace("```", "").strip()

        start = len(raw)
        for marker in ["#include", "__global__", "extern \"C\"", "using namespace std", "typedef", "struct"]:
            p = raw.find(marker)
            if p != -1 and p < start:
                start = p
        code = raw[start:] if start < len(raw) else raw
        lb = code.rfind("}")
        if lb != -1:
            code = code[:lb+1]
        code = code.strip()

        return (
            user_prompt
            + "[DECISION] PARALLELIZE [/DECISION]\n"
            + "[CUDA]\n" + code + "\n[/CUDA]\n"
            + "[LAUNCH]\n// optional\n[/LAUNCH]\n"
        )

    def masked_logprob(self, full_text: str) -> Tuple[torch.Tensor, Dict]:
        tok = self.tokenizer(full_text, return_tensors="pt")
        tok.pop("token_type_ids", None)
        if tok["input_ids"].shape[1] > MAX_LOGPROB_TOK:
            for k in tok:
                tok[k] = tok[k][:, -MAX_LOGPROB_TOK:]
        tok = {k: v.to("cuda:0") for k, v in tok.items()}
        logits = self.model(**tok).logits[0]
        ids = tok["input_ids"][0]

        def find_first(id_):
            pos = (ids == id_).nonzero(as_tuple=False)
            return int(pos[0]) if pos.numel() else None

        d_s = find_first(self.tid["[DECISION]"]); d_e = find_first(self.tid["[/DECISION]"])
        c_s = find_first(self.tid["[CUDA]"]);     c_e = find_first(self.tid["[/CUDA]"])
        l_s = find_first(self.tid["[LAUNCH]"]);   l_e = find_first(self.tid["[/LAUNCH]"])

        T = ids.shape[0]
        mask = torch.zeros(T, dtype=torch.bool, device=logits.device)

        def mark(s, e):
            if s is not None and e is not None and e > s + 1:
                mask[s+1:e] = True

        mark(d_s, d_e); mark(c_s, c_e); mark(l_s, l_e)

        logits = logits[:-1]
        targets = ids[1:]
        mask = mask[1:]

        logp = F.log_softmax(logits.float(), dim=-1)
        sel = logp[mask, targets[mask]]
        return sel.sum(), {"mask_count": int(mask.sum().item()), "len": int(T)}

def build_policy() -> Policy:
    tok = os.environ.get("HUGGINGFACE_TOKEN", "")
    if tok:
        login(token=tok)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.add_special_tokens(SPECIALS)

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
            max_memory={0: "38GiB", 1: "38GiB"},
            quantization_config=bnb_config,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=TRAIN_DTYPE,
            device_map="auto",
            max_memory={0: "38GiB", 1: "38GiB"},
        )
        model.resize_token_embeddings(len(tokenizer))

    wanted = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [w for w in wanted if w in present] or [w for w in ["q_proj","k_proj","v_proj","o_proj"] if w in present]
    print(f"[INFO] LoRA targets={targets}")

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", target_modules=targets, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return Policy(model, tokenizer)


# =========================
# Harness (BabelTower driver compile/run + signature pass_rate)
# =========================
def _run_cmd(cmd: List[str], timeout: float, env: Dict[str, str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        return -1, "", "TIMEOUT"
    return p.returncode, out, err

def _mk_common_driver_prelude() -> str:
    # Includes the fixed _print_arg overload set (avoids array/pointer ambiguity).
    return r"""
#include <bits/stdc++.h>
using namespace std;

static void _print_scalar(const float& v){ cout<<setprecision(8)<<v; }
static void _print_scalar(const double& v){ cout<<setprecision(16)<<v; }
template<class T> static void _print_scalar(const T& v){ cout<<v; }

// Scalars (non-array, non-pointer)
template<class T,
         typename std::enable_if<!std::is_array<T>::value && !std::is_pointer<T>::value, int>::type = 0>
static void _print_arg(T& x) { _print_scalar(x); }

// Arrays by reference
template<class T, size_t N>
static void _print_arg(T (&a)[N]) {
  cout << "[ ";
  for (size_t i = 0; i < N; i++) { _print_scalar(a[i]); if (i + 1 < N) cout << ", "; }
  cout << " ]";
}

// Pointers only (reference-to-pointer prevents array-decay ambiguity)
template<class T>
static void _print_arg(T* const& p) {
  if (!p) { cout << "[ ]"; return; }
  cout << "[ "; _print_scalar(p[0]); cout << " ]";
}

template <typename F, typename... Args>
static void wrapper(F f, Args&&... args) {
  using R = decltype(f(std::forward<Args>(args)...));
  if constexpr (std::is_void_v<R>) { f(std::forward<Args>(args)...); cout<<"Return value: void "; }
  else { R r=f(std::forward<Args>(args)...); cout<<"Return value: "; _print_scalar(r); cout<<" "; }
  cout<<"Arguments after function call: (";
  bool first=true;
  auto emit=[&](auto&& one){ if(!first) cout<<", "; first=false; _print_arg(one); };
  (emit(args), ...);
  cout<<")\n";
}
"""

def _build_cpp_program(cpp_code: str, cpp_tests: List[str]) -> str:
    return _mk_common_driver_prelude() + "\n" + cpp_code + "\nint main(){\n" + "\n".join(cpp_tests) + "\nreturn 0;}\n"

def _build_cuda_program(cuda_code: str, cuda_tests: List[str]) -> str:
    return _mk_common_driver_prelude() + "\n#include <cuda_runtime.h>\n" + cuda_code + "\nint main(){\n" + "\n".join(cuda_tests) + "\ncudaDeviceSynchronize();\nreturn 0;}\n"

def _extract_sigs(stdout: str) -> List[str]:
    sigs = []
    for ln in (stdout or "").splitlines():
        s = ln.strip()
        if "Return value:" in s and "Arguments after function call:" in s:
            sigs.append(s)
    return sigs

def _pass_rate(seq_sigs: List[str], cuda_sigs: List[str]) -> float:
    if not seq_sigs or not cuda_sigs:
        return 0.0
    n = min(len(seq_sigs), len(cuda_sigs))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if seq_sigs[i] == cuda_sigs[i]) / float(n)

@dataclass
class CandMetrics:
    compile_ok: bool
    pass_rate: float
    cpu_medians_ms: List[float]
    gpu_medians_ms: List[float]
    cv: float
    decision: str
    valid: bool
    compile_errs: List[str]
    runtime_errs: List[str]
    features: Dict[str, float]

class ExecutionHarness:
    def __init__(self, rank_gpu: int = 0,
                 gxx="g++", nvcc="nvcc",
                 cflags="-O3 -march=native -std=c++17",
                 nvccflags="-O3 -std=c++17",
                 trials=TRIALS_PER_INPUT,
                 timeout=RUN_TIMEOUT_SEC):
        self.rank_gpu = rank_gpu
        self.gxx = gxx
        self.nvcc = nvcc
        self.cflags = shlex.split(cflags)
        self.nvccflags = shlex.split(nvccflags)
        self.trials = max(1, int(trials))
        self.timeout = timeout

    def _env(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.rank_gpu)
        return env

    def _compile_c(self, src: str, out: str) -> Tuple[bool, str]:
        rc, outt, err = _run_cmd([self.gxx, src, "-o", out, *self.cflags], self.timeout, self._env())
        return (rc == 0, err)

    def _compile_cuda(self, src: str, out: str) -> Tuple[bool, str]:
        rc, outt, err = _run_cmd([self.nvcc, src, "-o", out, *self.nvccflags], self.timeout, self._env())
        return (rc == 0, err)

    def _time_run(self, exe: str) -> Tuple[int, str, str, float]:
        st = time.perf_counter()
        rc, outt, err = _run_cmd([exe], self.timeout, self._env())
        t_ms = (time.perf_counter() - st) * 1000.0
        return rc, outt, err, t_ms

    def compile_and_run_cpp_cached(
        self,
        td: str,
        task: Task,
        cpp_tests: List[str],
    ) -> Tuple[bool, str, Optional[float], List[str]]:
        cpp_src = os.path.join(td, "seq_prog.cpp")
        cpp_exe = os.path.join(td, "seq_prog.out")

        with open(cpp_src, "w", encoding="utf-8") as f:
            f.write(_build_cpp_program(task.cpp_code, cpp_tests))

        ok, err = self._compile_c(cpp_src, cpp_exe)
        if not ok:
            return False, (err or "")[:2000], None, []

        samples = []
        rc, out, er, t_ms = self._time_run(cpp_exe)
        if rc != 0:
            return False, ("SEQ RUNTIME ERROR:\n" + (er or ""))[:2000], None, []
        seq_sigs = _extract_sigs(out)
        samples.append(t_ms)

        for _ in range(max(1, self.trials) - 1):
            rc2, out2, er2, t2 = self._time_run(cpp_exe)
            if rc2 != 0:
                return False, ("SEQ RUNTIME ERROR:\n" + (er2 or ""))[:2000], None, []
            samples.append(t2)

        return True, "", float(_median_ms(samples)), seq_sigs

    def eval_one_cuda_candidate_cached_seq(
        self,
        td: str,
        full_text: str,
        task: Task,
        cuda_tests: List[str],
        seq_sigs: List[str],
    ) -> Dict[str, Any]:
        cuda_code = _extract_first(full_text, TAG_RX["CUDA_CODE"])
        decision_txt = _extract_first(full_text, TAG_RX["DECISION"]).upper()
        decision = "PARALLELIZE" if "PARALLEL" in decision_txt else "KEEP"

        out = {
            "decision": decision,
            "compile_ok": False,
            "pass_rate": 0.0,
            "correct": False,
            "cuda_med_ms": None,
            "compile_err": "",
            "runtime_err": "",
        }

        if decision != "PARALLELIZE" or not cuda_code.strip():
            return out

        cu_src = os.path.join(td, "cand_prog.cu")
        cu_exe = os.path.join(td, "cand_prog.out")
        with open(cu_src, "w", encoding="utf-8") as f:
            f.write(_build_cuda_program(cuda_code, cuda_tests))

        ok, err = self._compile_cuda(cu_src, cu_exe)
        out["compile_ok"] = bool(ok)
        if not ok:
            out["compile_err"] = (err or "")[:2000]
            return out

        samples = []
        rc, stdout, stderr, t_ms = self._time_run(cu_exe)
        if rc != 0:
            out["runtime_err"] = (stderr or "CUDA RUNTIME ERROR")[:2000]
            return out

        cuda_sigs = _extract_sigs(stdout)
        pr = _pass_rate(seq_sigs, cuda_sigs)

        samples.append(t_ms)
        for _ in range(max(1, self.trials) - 1):
            rc2, _, st2, t2 = self._time_run(cu_exe)
            if rc2 != 0:
                out["runtime_err"] = (st2 or "CUDA RUNTIME ERROR")[:2000]
                return out
            samples.append(t2)

        out["cuda_med_ms"] = float(_median_ms(samples))
        out["pass_rate"] = float(pr)
        out["correct"] = bool(pr >= 1.0)
        return out

    def run(self, full_text: str, task: Task) -> CandMetrics:
        decision_txt = _extract_first(full_text, TAG_RX["DECISION"]).upper()
        decision = "PARALLELIZE" if "PARALLEL" in decision_txt else "KEEP"
        cuda_code = _extract_first(full_text, TAG_RX["CUDA_CODE"])
        feats = extract_struct_features(cuda_code)

        cpp_tests = task.extra.get("consistent_cpp_inputs", []) or []
        cuda_tests = task.extra.get("consistent_cuda_inputs", []) or []

        compile_errs, runtime_errs = [], []

        if not cpp_tests or not cuda_tests:
            return CandMetrics(False, 0.0, [], [], 0.0, decision, False, ["MISSING TESTS"], [], feats)

        with tempfile.TemporaryDirectory() as td:
            ok_seq, seq_err, seq_med, seq_sigs = self.compile_and_run_cpp_cached(td, task, cpp_tests)
            if not ok_seq or seq_med is None:
                compile_errs.append("CPP COMPILE/RUN ERROR:\n" + (seq_err or "")[:2000])
                return CandMetrics(False, 0.0, [], [], 0.0, decision, False, compile_errs, runtime_errs, feats)

            if decision != "PARALLELIZE" or not cuda_code.strip():
                return CandMetrics(
                    False, 0.0, [float(seq_med)], [], 0.0, decision, False,
                    compile_errs + ["NO CUDA CODE"], runtime_errs, feats
                )

            cu_src = os.path.join(td, "prog.cu")
            cu_exe = os.path.join(td, "prog_cuda.out")
            with open(cu_src, "w", encoding="utf-8") as f:
                f.write(_build_cuda_program(cuda_code, cuda_tests))

            ok, err = self._compile_cuda(cu_src, cu_exe)
            if not ok:
                compile_errs.append("CUDA COMPILE ERROR:\n" + (err or "")[:2000])
                return CandMetrics(False, 0.0, [float(seq_med)], [], 0.0, decision, False, compile_errs, runtime_errs, feats)

            cu_samples = []
            rc1, out1, err1, t1 = self._time_run(cu_exe)
            if rc1 != 0:
                runtime_errs.append("CUDA RUNTIME ERROR:\n" + (err1 or "")[:2000])
                return CandMetrics(False, 0.0, [float(seq_med)], [t1], 0.0, decision, False, compile_errs, runtime_errs, feats)

            cuda_sigs = _extract_sigs(out1)
            cu_samples.append(t1)
            for _ in range(self.trials - 1):
                _, _, _, t = self._time_run(cu_exe)
                cu_samples.append(t)

            pr = _pass_rate(seq_sigs, cuda_sigs)
            gpu_med = float(_median_ms(cu_samples))
            cv = float(_cv(cu_samples))  # seq cv is baked into seq run; keep simple here
            valid = bool(pr > 0.0)

            return CandMetrics(True, float(pr), [float(seq_med)], [gpu_med], cv, decision, valid, compile_errs, runtime_errs, feats)


# =========================
# Reward + GRPO
# =========================
def reward_from_metrics(m: CandMetrics) -> float:
    if not m.compile_ok:
        return -PENAL_FAIL_COMPILE

    correctness_penalty = -PENAL_FAIL_WRONG * (1.0 - m.pass_rate)
    r_struct = structure_reward(m.features) if USE_STRUCT_RANKER else 0.0

    if m.pass_rate <= 0.0:
        return float(correctness_penalty + STRUCT_COEF * r_struct)

    if m.pass_rate >= 1.0 and m.cpu_medians_ms and m.gpu_medians_ms:
        S = max(m.cpu_medians_ms[0] / max(m.gpu_medians_ms[0], 1e-6), 1e-3)
        r_speed = ALPHA_SPEED * float(torch.clamp(torch.tensor(math.log2(S)), *CLIP_LOG2_S))
        return float(correctness_penalty + r_speed + STRUCT_COEF * r_struct - PENAL_CV * m.cv)

    return float(correctness_penalty + BONUS_KEEP_GOOD * m.pass_rate + STRUCT_COEF * r_struct - PENAL_CV * m.cv)

def grpo_loss_backward(policy: Policy, texts: List[str], old_logps: torch.Tensor, rewards: torch.Tensor) -> float:
    total = 0.0
    for i, t in enumerate(texts):
        new_lp, _ = policy.masked_logprob(t)
        ratio = torch.exp(new_lp - old_logps[i])
        adv = rewards[i]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
        loss = -torch.min(unclipped, clipped)
        loss.backward()
        total += float(loss.detach().item())
    return total / max(1, len(texts))


# =========================
# Self-refinement
# =========================
def build_feedback(metrics: CandMetrics) -> str:
    if not metrics.compile_ok:
        msg = ["Previous CUDA failed compile/run."]
        for e in metrics.compile_errs[:1]:
            msg.append(e[:800])
        for e in metrics.runtime_errs[:1]:
            msg.append(e[:800])
        return "\n".join(msg)
    if metrics.pass_rate < 1.0:
        return f"CUDA ran but only {metrics.pass_rate*100:.1f}% of tests matched signatures."
    return "CUDA is correct on all tests."

def generate_population_and_refine(policy: Policy, harness: ExecutionHarness, task: Task, fewshot_pool: None, fewshot_k: 0, fewshot_max_chars: 12000) -> Tuple[List[str], List[CandMetrics]]:

    texts, mets = [], []
    base = task_to_prompt(task, fewshot_pool=fewshot_pool, fewshot_k=fewshot_k, fewshot_max_chars=fewshot_max_chars)

    for i in range(max(1, N_INIT_VARIANTS)):
        temp = TEMPS[i % len(TEMPS)]
        full = policy.generate_candidate(base, temperature=temp, top_p=TOP_P)
        texts.append(full)
        mets.append(harness.run(full, task))

    rewards = [reward_from_metrics(m) for m in mets]
    best_i = max(range(len(rewards)), key=lambda i: rewards[i])
    prev_full = texts[best_i]
    prev_m = mets[best_i]

    for r in range(max(0, N_REFINE_STEPS)):
        fb = build_feedback(prev_m)
        prev_cuda = _extract_first(prev_full, TAG_RX["CUDA_CODE"])
        refined_prompt = (
            base
            + "\n\n[PREV_CUDA]\n" + (prev_cuda if prev_cuda else "") + "\n[/PREV_CUDA]\n"
            + "Issues:\n" + fb + "\n\n"
            + "Generate a NEW corrected CUDA C++ program. Only output CUDA code.\n"
        )
        if USE_NO_THINK:
            refined_prompt += "\n/no_think\n"
        temp = TEMPS[(N_INIT_VARIANTS + r) % len(TEMPS)]
        full = policy.generate_candidate(refined_prompt, temperature=temp, top_p=TOP_P)
        m = harness.run(full, task)
        texts.append(full)
        mets.append(m)
        prev_full, prev_m = full, m
        if m.compile_ok and m.pass_rate >= 1.0:
            break

    return texts, mets


# =========================
# One RLAIF/GRPO step
# =========================
def rlaif_step(policy: Policy, harness: ExecutionHarness, task: Task, fewshot_pool: None, fewshot_k: 0, fewshot_max_chars: 12000) -> Dict[str, float]:
    texts, mets = generate_population_and_refine(policy, harness, task, fewshot_pool, fewshot_k, fewshot_max_chars)
    with torch.no_grad():
        old = torch.stack([policy.masked_logprob(t)[0] for t in texts]).to("cuda:0")
    rewards = torch.tensor([reward_from_metrics(m) for m in mets], device="cuda:0", dtype=torch.float32)
    loss = grpo_loss_backward(policy, texts, old, rewards)
    best_pass = max((m.pass_rate for m in mets), default=0.0)
    return {
        "loss_grpo": float(loss),
        "rewards_mean": float(rewards.mean().item()),
        "best_reward": float(rewards.max().item()),
        "best_pass": float(best_pass),
    }


# =========================
# Export
# =========================
def export_task_result(out_dir: str, task: Task, best_full_text: str, best_m: CandMetrics, reward: float):
    os.makedirs(out_dir, exist_ok=True)
    cuda_code = _extract_first(best_full_text, TAG_RX["CUDA_CODE"])
    sp = compute_speedups(best_m.cpu_medians_ms, best_m.gpu_medians_ms)
    payload = {
        "task_id": task.task_id,
        "reward": float(reward),
        "compile_ok": bool(best_m.compile_ok),
        "pass_rate": float(best_m.pass_rate),
        "cv": float(best_m.cv),
        **sp,
        "cpp_code": task.cpp_code,
        "cuda_code": cuda_code,
        "features": best_m.features,
        "compile_errs": best_m.compile_errs,
        "runtime_errs": best_m.runtime_errs,
    }
    path = os.path.join(out_dir, f"{task.task_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[EXPORT] {path}")


# =========================
# Final evaluation (pass@k + speedup vs sequential)
# =========================
@torch.no_grad()
def final_evaluate_babeltower(policy: Policy, harness: ExecutionHarness, tasks: List[Task], fewshot_pool: None, fewshot_k: 0, fewshot_max_chars: 12000) -> None:
    print("\n====================")
    print("Running final BabelTower evaluation with pass@k + speedup vs sequential ...")
    print("====================\n")

    policy.eval()

    K_LIST = [1, 3, 5, 10]
    MAX_K = max(K_LIST)

    eval_top_p = TOP_P
    eval_temps = TEMPS[:] if TEMPS else [0.2]

    agg = {
        k: {
            "n_tasks": 0,
            "pass_compile": 0,
            "pass_test": 0,
            "pass_speedup_gt1": 0,
            "sum_best_speedup_uncond": 0.0,
            "sum_best_speedup_cond": 0.0,
            "n_best_speedup_cond": 0,
        }
        for k in K_LIST
    }

    def _speedup(seq_ms: float, cuda_ms: float) -> float:
        seq_ms = max(_safe_float(seq_ms, float("inf")), 1e-9)
        cuda_ms = max(_safe_float(cuda_ms, float("inf")), 1e-9)
        return seq_ms / cuda_ms

    for task in tasks:
        cpp_tests = task.extra.get("consistent_cpp_inputs", []) or []
        cuda_tests = task.extra.get("consistent_cuda_inputs", []) or []
        if not cpp_tests or not cuda_tests:
            continue

        user_prompt = task_to_prompt(task, fewshot_pool=fewshot_pool, fewshot_k=fewshot_k, fewshot_max_chars=fewshot_max_chars)

        with tempfile.TemporaryDirectory() as td:
            ok_seq, seq_err, seq_med_ms, seq_sigs = harness.compile_and_run_cpp_cached(td, task, cpp_tests)
            if not ok_seq or seq_med_ms is None or not seq_sigs:
                continue

            cand_results: List[Dict[str, Any]] = []
            for i in range(MAX_K):
                temp = eval_temps[i % len(eval_temps)]
                full_text = policy.generate_candidate(user_prompt, temperature=temp, top_p=eval_top_p)
                r = harness.eval_one_cuda_candidate_cached_seq(td, full_text, task, cuda_tests, seq_sigs)
                if r["correct"] and r["cuda_med_ms"] is not None:
                    r["speedup"] = _speedup(seq_med_ms, r["cuda_med_ms"])
                else:
                    r["speedup"] = 0.0
                cand_results.append(r)

            for k in K_LIST:
                firstk = cand_results[:k]
                any_compile = any(r["compile_ok"] for r in firstk)
                any_correct = any(r["correct"] for r in firstk)
                any_speedup_gt1 = any((r["correct"] and r["speedup"] > 1.0) for r in firstk)

                best_speedup = 0.0
                if any_correct:
                    best_speedup = max(r["speedup"] for r in firstk if r["correct"])

                agg[k]["n_tasks"] += 1
                agg[k]["pass_compile"] += 1 if any_compile else 0
                agg[k]["pass_test"] += 1 if any_correct else 0
                agg[k]["pass_speedup_gt1"] += 1 if any_speedup_gt1 else 0
                agg[k]["sum_best_speedup_uncond"] += float(best_speedup)
                if any_correct:
                    agg[k]["sum_best_speedup_cond"] += float(best_speedup)
                    agg[k]["n_best_speedup_cond"] += 1

    print("\n==================== FINAL BABELTOWER PASS@K EVAL REPORT ====================")
    print(f"Model: {MODEL_NAME}")
    n_total = agg[1]["n_tasks"]
    print(f"Tasks evaluated (after filtering seq-compile+run): {n_total}")
    print()

    for k in [1, 3, 5]:
        n = max(1, agg[k]["n_tasks"])
        p_compile = agg[k]["pass_compile"] / n
        p_test = agg[k]["pass_test"] / n
        p_su1 = agg[k]["pass_speedup_gt1"] / n
        avg_best_su_uncond = agg[k]["sum_best_speedup_uncond"] / n
        denom = max(1, agg[k]["n_best_speedup_cond"])
        avg_best_su_cond = agg[k]["sum_best_speedup_cond"] / denom

        print(f"pass@{k}:")
        print(f"  compilation         : {p_compile:.3f}")
        print(f"  test-case pass      : {p_test:.3f}")
        print(f"  speedup>1 (correct) : {p_su1:.3f}")
        print(f"  avg best speedup (uncond, 0 if none correct): {avg_best_su_uncond:.3f}")
        print(f"  avg best speedup (cond on any correct)      : {avg_best_su_cond:.3f}  (n={agg[k]['n_best_speedup_cond']})")
        print()

    print("==========================================================================\n")
    policy.train()


# =========================
# Save policy
# =========================
def save_policy(policy: Policy, save_dir: str, save_tokenizer: bool = True):
    os.makedirs(save_dir, exist_ok=True)
    print(f"[SAVE] LoRA adapters -> {save_dir}")
    policy.model.save_pretrained(save_dir)
    if save_tokenizer:
        print(f"[SAVE] tokenizer -> {save_dir}")
        policy.tokenizer.save_pretrained(save_dir)


# =========================
# Main
# =========================
def main():
    global TOTAL_STEPS, EPOCHS, N_INIT_VARIANTS, N_REFINE_STEPS, MAX_STEP_TOK, MAX_LOGPROB_TOK
    global USE_STRUCT_RANKER, RANKER_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="test")
    ap.add_argument("--train_limit", type=int, default=-1, help="limit training tasks (debug); -1 = no limit")
    ap.add_argument("--eval_limit", type=int, default=200, help="limit eval tasks; keep sane (default 200)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--n_init", type=int, default=N_INIT_VARIANTS)
    ap.add_argument("--n_refine", type=int, default=N_REFINE_STEPS)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_STEP_TOK)
    ap.add_argument("--max_logprob_tok", type=int, default=MAX_LOGPROB_TOK)
    ap.add_argument("--save_dir", type=str, default="checkpoints/qwen3-32b-qlora-grpo-babeltower-full")
    ap.add_argument("--export_dir", type=str, default="optimized_cuda_outputs_babeltower_full")
    ap.add_argument("--no_export", action="store_true")
    ap.add_argument("--no_struct_ranker", action="store_true")
    ap.add_argument("--ranker_path", type=str, default=RANKER_PATH)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fewshot_k", type=int, default=2)
    ap.add_argument("--fewshot_max_chars", type=int, default=12000)
    ap.add_argument("--fewshot_seed", type=int, default=123)    
    args = ap.parse_args()

    TOTAL_STEPS = int(args.total_steps)
    EPOCHS = int(args.epochs)
    N_INIT_VARIANTS = int(args.n_init)
    N_REFINE_STEPS = int(args.n_refine)
    MAX_STEP_TOK = int(args.max_new_tokens)
    MAX_LOGPROB_TOK = int(args.max_logprob_tok)

    if args.no_struct_ranker:
        USE_STRUCT_RANKER = False
    else:
        RANKER_PATH = args.ranker_path

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if USE_STRUCT_RANKER:
        _load_ranker_if_any(RANKER_PATH)

    train_limit = None if args.train_limit is None or args.train_limit < 0 else int(args.train_limit)
    eval_limit = None if args.eval_limit is None or args.eval_limit < 0 else int(args.eval_limit)

    # --- Use the SAME 233 tasks for training and evaluation ---
    

    tasks_233 = load_tasks_from_babeltower(args.train_split, limit=ALL_TASKS_LIMIT)
    
    fewshot_pool = FewshotPool(tasks_233, seed=args.fewshot_seed)
    print(f"[FEWSHOT] pool_size={len(fewshot_pool.examples)} / {len(tasks_233)} (tasks with cuda_code/wrapper)")

    # If you want reproducible order (highly recommended):
    random.seed(args.seed)
    random.shuffle(tasks_233)

    train_tasks = tasks_233
    eval_tasks = tasks_233

    # print(f"[DATA] using same tasks for train+eval: {len(tasks_233)}")

    print(f"[DATA] train_tasks={len(train_tasks)} (split={args.train_split}, limit={train_limit})")
    print(f"[DATA] eval_tasks={len(eval_tasks)} (split={args.eval_split}, limit={eval_limit})")

    if torch.cuda.device_count() < 2:
        print(f"[WARN] torch sees only {torch.cuda.device_count()} CUDA device(s). Expected 2 for 2×A100 sharding.")

    policy = build_policy()
    policy.train()

    if HAVE_BNB:
        optimizer = bnb.optim.PagedAdamW8bit(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, TOTAL_STEPS)

    harness = ExecutionHarness(rank_gpu=0)

    step = 0
    for epoch in range(max(1, EPOCHS)):
        random.shuffle(train_tasks)
        for task in train_tasks:
            if step >= TOTAL_STEPS:
                break

            optimizer.zero_grad(set_to_none=True)
            out = rlaif_step(policy, harness, task, fewshot_pool, args.fewshot_k, args.fewshot_max_chars)
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % PRINT_EVERY == 0:
                print(
                    f"[{step}] loss={out['loss_grpo']:.3f} "
                    f"Rmean={out['rewards_mean']:.3f} bestR={out['best_reward']:.3f} "
                    f"best_pass={out['best_pass']:.2f} task={task.task_id}"
                )
            step += 1

        if step >= TOTAL_STEPS:
            break

    print("[TRAIN] finished.")

    # ---- Eval (pass@k + speedup) ----
    final_evaluate_babeltower(policy, harness, eval_tasks, fewshot_pool, args.fewshot_k, args.fewshot_max_chars)

    # ---- Export best per eval task (expensive; default ON but you can disable with --no_export) ----
    if not args.no_export:
        print(f"[EXPORT] exporting best CUDA for eval tasks -> {args.export_dir}")
        policy.eval()
        for task in eval_tasks:
            texts, mets = generate_population_and_refine(policy, harness, task, fewshot_pool, args.fewshot_k, args.fewshot_max_chars)
            rewards = [reward_from_metrics(m) for m in mets]
            bi = max(range(len(rewards)), key=lambda i: rewards[i])
            export_task_result(args.export_dir, task, texts[bi], mets[bi], rewards[bi])

    # ---- Save adapters ----
    save_policy(policy, args.save_dir, save_tokenizer=SAVE_TOKENIZER)
    print("[DONE] full run finished.")


if __name__ == "__main__":
    main()
