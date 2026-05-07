#!/usr/bin/env python3
# Works for: 1× GPU (or 2× GPUs) with QLoRA (4-bit) + LoRA + GRPO on Qwen3-8B
#
# Install (example):
#   pip install -U "transformers>=4.45.0" peft accelerate bitsandbytes huggingface_hub torch numpy
#
# Run (single GPU recommended for 8B):
#   export CUDA_VISIBLE_DEVICES=0
#   export HUGGINGFACE_TOKEN=...
#   python train_qwen3_8b_qlora_grpo.py
#
# Notes:
# - You can still run on 2 GPUs with device_map="auto", but 8B usually fits well on 1 GPU.
# - Harness pins compiled CUDA executions to GPU0 (rank_gpu=0).
# - DPO/KL disabled by default to save memory (no ref model loaded).

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
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable

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

# Try bitsandbytes optimizer
try:
    import bitsandbytes as bnb
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False


SAVE_DIR = "checkpoints/qwen3-8b-qlora-grpo-full-dataset-samples-par-features-write-output"
SAVE_TOKENIZER = True
EXPORT_DIR = "optimized_cuda_outputs_json"

# =========================
# 0) Helpers for CUDA WRITE
# =========================

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def compute_speedups(cpu_medians_ms: List[float], gpu_medians_ms: List[float]) -> Dict:
    n = min(len(cpu_medians_ms), len(gpu_medians_ms))
    per_input = []
    for i in range(n):
        cpu = max(_safe_float(cpu_medians_ms[i], float("inf")), 1e-9)
        gpu = max(_safe_float(gpu_medians_ms[i], float("inf")), 1e-9)
        per_input.append(cpu / gpu)
    speedup_first = per_input[0] if per_input else 0.0
    speedup_mean = (sum(per_input) / len(per_input)) if per_input else 0.0
    return {
        "speedup_per_input": per_input,
        "speedup_first": speedup_first,
        "speedup_mean": speedup_mean,
    }


# =========================
# 0) CONFIG (Qwen3-8B)
# =========================
MODEL_NAME = "Qwen/Qwen3-8B"       # <-- Qwen3 base 8B
USE_4BIT_QLORA = True
TRAIN_DTYPE = torch.bfloat16

SYSTEM_PROMPT = "You are an expert CUDA C++ engineer."

# Qwen3 has /think and /no_think toggles. /no_think often reduces verbose outputs.
USE_NO_THINK = True

# Exploration + refinement
N_INIT_VARIANTS = 4
N_REFINE_STEPS = 3

TEMPS = [0.2, 0.7]
TOP_P = 0.9

# For 8B you can often increase these a bit, but keep sane.
MAX_STEP_TOK = 1536
MAX_LOGPROB_TOK = 1024

# GRPO params
LAMBDA_GRPO = 1.0
CLIP_EPS = 0.2

# Disable DPO/KL entirely to save memory (no ref model)
LAMBDA_DPO = 0.0
BETA_KL = 0.0

# Reward params
ALPHA_SPEED     = 1.0
CLIP_LOG2_S     = (-0.5, 4.0)
PENAL_FAIL_COMPILE = 3.0
PENAL_FAIL_WRONG   = 2.0
PENAL_CV        = 0.5
BONUS_KEEP_GOOD = 0.5

# Optim (8B can usually tolerate a slightly higher LR than 32B; keep conservative)
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
TOTAL_STEPS  = 10_000

# Dataset path
JSONL_PATH = "output_jsonl/sft-ft.jsonl"

# Harness
TRIALS_PER_INPUT = 3
WARMUP_RUNS = 1
RUN_TIMEOUT_SEC = 240.0

# Few-shot context (from dataset CUDA samples with speedup > 1)
USE_FEWSHOT_CONTEXT = True
FEWSHOT_K = 2
FEWSHOT_MIN_SPEEDUP = 1.01
FEWSHOT_MAX_CHARS_PER_CODE = 8000
FEWSHOT_SEED = 1337


# =========================
# STRUCTURE FEATURE EXTRACTOR (static proxies)
# =========================

with open("struct_ranker-64-min-speedup-1-epoch-150.json") as f:
    RANKER = json.load(f)

W_f = torch.tensor(RANKER["weights"], device="cuda:0")
MEAN_f = torch.tensor(RANKER["norm_mean"], device="cuda:0")
STD_f  = torch.tensor(RANKER["norm_std"],  device="cuda:0")

def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//.*?$", "", src, flags=re.MULTILINE)
    return src

KERNEL_DEF_RX = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*\{", re.DOTALL)
ARRAY_SUB_RX = re.compile(r"\b([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]")
ASSIGN_RX    = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);")

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
    out: List[Tuple[str, str]] = []
    for m in KERNEL_DEF_RX.finditer(src):
        name = m.group(1)
        body_start = m.end() - 1  # points to '{'
        body_end = _find_matching_brace(src, body_start)
        if body_end == -1:
            continue
        body = src[body_start:body_end + 1]
        out.append((name, body))
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
    plus = len(re.findall(r"\+", b))
    minus = len(re.findall(r"-", b))
    mult = len(re.findall(r"\*", b))
    div  = len(re.findall(r"/", b))
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
    if not cuda_src or not cuda_src.strip():
        return {
            "coal": 0.0, "ai": 0.0, "occ": 0.0, "div": 0.0,
            "xfer": 0.0, "atomics": 0.0, "syncthreads": 0.0,
            "kernels": 0.0, "tpb": 0.0, "gmem_access": 0.0, "ops": 0.0
        }

    src = _strip_comments(cuda_src)
    kernels = _extract_kernels(src)
    tpb = _parse_threads_per_block(src)
    xfer = _transfer_proxy(src)

    coal_list: List[float] = []
    ai_list:   List[float] = []
    occ_list:  List[float] = []
    div_list:  List[float] = []

    atomics_total = 0
    sync_total = 0
    gmem_total = 0
    ops_total = 0

    for _, body in kernels:
        sym = _build_symtab(body)

        mem_scores: List[float] = []
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
        syn  = _count_syncthreads(body)

        atomics_total += atom
        sync_total += syn
        gmem_total += gmem
        ops_total += ops

        coal_list.append(coal_k)
        ai_list.append(ai_k)
        occ_list.append(occ_k)
        div_list.append(div_k)

    def avg(xs: List[float]) -> float:
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


# =========================
# 1) DATA: Task spec
# =========================
@dataclass
class Task:
    task_id: str
    seq_code: str
    inputs_text: str
    hardware_text: str
    extra: Dict = None

@dataclass
class SeedCandidate:
    task_id: str
    decision: str
    cuda_code: str
    launch_text: str
    meta: Dict

@dataclass
class FewshotExample:
    task_id: str
    speedup: float
    seq_code: str
    cuda_code: str
    inputs_text: str
    hardware_text: str
    meta: Dict


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _normalize_decision(d: Optional[str]) -> str:
    if not d:
        return "KEEP"
    d = d.strip().upper()
    if d in ("KEEP", "PARALLEL", "PARALLELIZE", "PARALLELISE"):
        return "PARALLELIZE" if d != "KEEP" else "KEEP"
    return "KEEP"

def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n// ... [TRUNCATED]\n"

def _meta_speedup_from_rec(meta: Dict) -> Optional[float]:
    if not isinstance(meta, dict):
        return None
    CAND_KEYS = [
        "gt_speedup",
        "ground_truth_speedup",
        "speedup",
        "avg_speedup",
        "best_speedup",
        "cuda_speedup",
        "oracle_speedup",
        "speedup_gt",
        "max_speedup",
    ]
    for k in CAND_KEYS:
        if k in meta:
            try:
                v = float(meta[k])
                if math.isfinite(v) and v > 0.0:
                    return v
            except Exception:
                pass
    for nest in ["metrics", "perf", "performance", "eval"]:
        sub = meta.get(nest)
        if isinstance(sub, dict):
            for k in CAND_KEYS:
                if k in sub:
                    try:
                        v = float(sub[k])
                        if math.isfinite(v) and v > 0.0:
                            return v
                    except Exception:
                        pass
    return None

def load_tasks_and_seeds_from_jsonl(
    jsonl_path: str,
    drop_bad_parallel: bool = True,
) -> Tuple[List[Task], List[SeedCandidate], List[FewshotExample]]:
    tasks: List[Task] = []
    seeds: List[SeedCandidate] = []
    fewshot_pool: List[FewshotExample] = []

    idx = 0
    for rec in _iter_jsonl(jsonl_path):
        idx += 1
        task_id = rec.get("task_id") or f"jsonl_{idx:06d}"
        seq_code = (rec.get("seq_code") or "").strip()
        if not seq_code:
            continue

        inputs_text = (rec.get("inputs_text") or "").strip()
        hw_text = (rec.get("hardware_text") or "").strip()
        decision = _normalize_decision(rec.get("decision"))
        cuda_code = (rec.get("cuda_code") or "").strip()
        launch_text = (rec.get("launch_text") or "").strip()
        meta = rec.get("meta") or {}

        tasks.append(Task(
            task_id=task_id,
            seq_code=seq_code,
            inputs_text=inputs_text,
            hardware_text=hw_text or "Unknown",
            extra={"decision": decision, "meta": meta},
        ))

        if decision == "KEEP":
            seeds.append(SeedCandidate(task_id, "KEEP", "", "", meta))
        else:
            if cuda_code:
                seeds.append(SeedCandidate(task_id, "PARALLELIZE", cuda_code, launch_text, meta))
            else:
                if not drop_bad_parallel:
                    seeds.append(SeedCandidate(
                        task_id, "KEEP", "", "",
                        {**meta, "note": "demoted: parallel missing cuda"},
                    ))

        sp = _meta_speedup_from_rec(meta)
        if (
            decision == "PARALLELIZE"
            and cuda_code
            and sp is not None
            and sp >= FEWSHOT_MIN_SPEEDUP
        ):
            fewshot_pool.append(FewshotExample(
                task_id=task_id,
                speedup=float(sp),
                seq_code=seq_code,
                cuda_code=cuda_code,
                inputs_text=inputs_text,
                hardware_text=hw_text or "Unknown",
                meta=meta,
            ))

    return tasks, seeds, fewshot_pool


def build_fewshot_context(
    pool: List[FewshotExample],
    current_task_id: str,
    k: int,
) -> str:
    if not USE_FEWSHOT_CONTEXT or k <= 0 or not pool:
        return ""

    cand = [ex for ex in pool if ex.task_id != current_task_id]
    if not cand:
        return ""

    cand.sort(key=lambda ex: ex.speedup, reverse=True)
    topN = min(len(cand), max(20, k * 10))
    cand = cand[:topN]

    random.shuffle(cand)
    chosen = cand[:k]

    blocks = ["[FEWSHOT]"]
    for ex in chosen:
        blocks.append("[EXAMPLE]")
        blocks.append(f"// dataset_speedup≈{ex.speedup:.3f}x  task_id={ex.task_id}")
        blocks.append("[SEQ_CODE]")
        blocks.append(_truncate(ex.seq_code, FEWSHOT_MAX_CHARS_PER_CODE))
        blocks.append("[/SEQ_CODE]")
        blocks.append("[CUDA]")
        blocks.append(_truncate(ex.cuda_code, FEWSHOT_MAX_CHARS_PER_CODE))
        blocks.append("[/CUDA]")
        if ex.inputs_text:
            blocks.append("[INPUTS]")
            blocks.append(_truncate(ex.inputs_text, 2000))
            blocks.append("[/INPUTS]")
        if ex.hardware_text:
            blocks.append("[HARDWARE]")
            blocks.append(_truncate(ex.hardware_text, 2000))
            blocks.append("[/HARDWARE]")
        blocks.append("[/EXAMPLE]")
        blocks.append("")
    blocks.append("[/FEWSHOT]")
    blocks.append("")
    return "\n".join(blocks)


def task_to_prompt(task: Task, fewshot_pool: Optional[List[FewshotExample]] = None) -> str:
    fewshot = ""
    if fewshot_pool:
        fewshot = build_fewshot_context(fewshot_pool, current_task_id=task.task_id, k=FEWSHOT_K)

    user = (
        fewshot
        + "Convert the following C++ program into an efficient CUDA C++ program "
          "that produces the same output on the given hardware.\n"
          "Only output CUDA C++ code that can be compiled by nvcc as a single translation unit.\n"
          "Do not include explanations, comments about what you are doing, or any markdown fences.\n\n"
        f"[SEQ_CODE]\n{task.seq_code}\n[/SEQ_CODE]\n"
        f"[INPUTS]\n{task.inputs_text}\n[/INPUTS]\n"
        f"[HARDWARE]\n{task.hardware_text}\n[/HARDWARE]\n"
    )

    if USE_NO_THINK:
        user += "\n/no_think\n"
    return user


# =========================
# 2) TOKEN TAGS + POLICY
# =========================
SPECIALS = {
    "additional_special_tokens": [
        "[SEQ_CODE]","[/SEQ_CODE]",
        "[INPUTS]","[/INPUTS]",
        "[HARDWARE]","[/HARDWARE]",
        "[DECISION]","[/DECISION]",
        "[CUDA]","[/CUDA]",
        "[LAUNCH]","[/LAUNCH]",
        "<NO_CUDA>",
        "[PREV_CUDA]","[/PREV_CUDA]",
        "[FEWSHOT]","[/FEWSHOT]",
        "[EXAMPLE]","[/EXAMPLE]",
    ]
}

TAG_RX = {
    "SEQ_CODE":   re.compile(r"\[SEQ_CODE\](.*?)\[/SEQ_CODE\]", re.DOTALL),
    "CUDA_CODE":  re.compile(r"\[CUDA\](.*?)\[/CUDA\]",         re.DOTALL),
    "DECISION":   re.compile(r"\[DECISION\](.*?)\[/DECISION\]", re.DOTALL),
}

def _extract_first(text: str, rx: re.Pattern) -> str:
    m = rx.search(text)
    return (m.group(1).strip() if m else "").strip()


class Policy(nn.Module):
    """
    Single-process wrapper around a (possibly sharded) causal LM.
    IMPORTANT:
      - Do NOT .to(device) the model; it may be sharded via device_map="auto".
      - Put small input tensors on cuda:0; HF dispatches internally.
    """
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
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
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
            pad_token_id=self.tokenizer.eos_token_id,
        )

        out = self.model.generate(**tok, generation_config=gen_cfg)
        gen_ids = out[0][input_len:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

        raw = (
            raw.replace("```cpp", "")
               .replace("```c++", "")
               .replace("```cu", "")
               .replace("```cuda", "")
               .replace("```", "")
        ).strip()

        start_idx = len(raw)
        for marker in ["#include", "__global__", "int main", "using namespace std"]:
            pos = raw.find(marker)
            if pos != -1 and pos < start_idx:
                start_idx = pos
        code = raw[start_idx:] if start_idx < len(raw) else raw

        last_brace = code.rfind("}")
        if last_brace != -1:
            code = code[: last_brace + 1]

        code = code.strip()

        full_text = (
            user_prompt
            + "[DECISION] PARALLELIZE [/DECISION]\n"
            + "[CUDA]\n"
            + code
            + "\n[/CUDA]\n"
            + "[LAUNCH]\n"
            + "// launch configuration (optional)\n"
            + "[/LAUNCH]\n"
        )
        return full_text

    def masked_logprob(self, full_text: str) -> Tuple[torch.Tensor, Dict]:
        tok = self.tokenizer(full_text, return_tensors="pt")
        tok.pop("token_type_ids", None)
        input_ids = tok["input_ids"][0]

        if input_ids.shape[0] > MAX_LOGPROB_TOK:
            for k in tok:
                tok[k] = tok[k][:, -MAX_LOGPROB_TOK:]
            input_ids = tok["input_ids"][0]

        tok = {k: v.to("cuda:0") for k, v in tok.items()}
        logits = self.model(**tok).logits[0]
        input_ids = tok["input_ids"][0]

        def find_first(id_):
            pos = (input_ids == id_).nonzero(as_tuple=False)
            return int(pos[0]) if pos.numel() else None

        d_s = find_first(self.tid["[DECISION]"]); d_e = find_first(self.tid["[/DECISION]"])
        c_s = find_first(self.tid["[CUDA]"]);     c_e = find_first(self.tid["[/CUDA]"])
        l_s = find_first(self.tid["[LAUNCH]"]);   l_e = find_first(self.tid["[/LAUNCH]"])

        T = input_ids.shape[0]
        mask = torch.zeros(T, dtype=torch.bool, device=logits.device)

        def mark(s, e):
            if s is not None and e is not None and e > s + 1:
                mask[s+1:e] = True

        mark(d_s, d_e)
        mark(c_s, c_e)
        mark(l_s, l_e)

        logits = logits[:-1]
        targets = input_ids[1:]
        mask = mask[1:]

        logp = F.log_softmax(logits.float(), dim=-1)
        sel = logp[mask, targets[mask]]
        total_logprob = sel.sum()
        info = {"mask_count": int(mask.sum().item()), "len": int(T)}
        return total_logprob, info


# =========================
# 3) HARNESS (compile/run → reward)
# =========================
@dataclass
class CandMetrics:
    compile_ok: bool
    pass_rate: float
    cpu_medians_ms: List[float]
    gpu_medians_ms: List[float]
    cv: float
    decision: str
    valid: bool
    seq_outputs: Optional[List[str]] = None
    cuda_outputs: Optional[List[str]] = None
    compile_errs: Optional[List[str]] = None
    features: Optional[Dict[str, float]] = None


def _parse_inputs(inputs_text: str) -> List[List[str]]:
    if not inputs_text:
        return []
    chunks = [c.strip() for c in inputs_text.split(";") if c.strip()]
    out: List[List[str]] = []
    for ch in chunks:
        rest = ch.split(":", 1)[-1] if ":" in ch else ch
        args = [a for a in rest.strip().split() if a]
        if args:
            out.append(args)
    return out

def _median_ms(samples: List[float]) -> float:
    return float(statistics.median(samples)) if samples else float("inf")

def _cv(samples: List[float]) -> float:
    if not samples:
        return 0.0
    m = statistics.mean(samples)
    if m == 0:
        return 0.0
    s = statistics.pstdev(samples)
    return float(s / m)

def _compute_pass_rate(seq_outputs: List[str], cuda_outputs: List[str],
                       atol: float = 1e-6, rtol: float = 1e-6) -> float:
    if not seq_outputs or not cuda_outputs:
        return 0.0
    total = min(len(seq_outputs), len(cuda_outputs))
    if total == 0:
        return 0.0
    matches = 0
    for s_out, c_out in zip(seq_outputs, cuda_outputs):
        s = s_out.strip()
        c = c_out.strip()
        # try numeric compare on first token
        try:
            s_val = float(s.split()[0])
            c_val = float(c.split()[0])
            if abs(s_val - c_val) <= atol + rtol * max(abs(s_val), 1.0):
                matches += 1
        except Exception:
            if s == c:
                matches += 1
    return matches / float(total)

def _run_cmd(
    cmd: List[str],
    timeout: float,
    extra_env: Optional[Dict[str, str]] = None,
    stdin_data: Optional[str] = None,
) -> Tuple[int, str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        out, err = p.communicate(input=stdin_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        return -1, "", "TIMEOUT"
    return p.returncode, out, err

def _outputs_match(seq_out: str, cuda_out: str, atol: float = 1e-6) -> bool:
    s = seq_out.strip()
    c = cuda_out.strip()
    if not s and not c:
        return True
    try:
        s_val = float(s.split()[-1])
        c_val = float(c.split()[-1])
        return abs(s_val - c_val) <= atol
    except Exception:
        return s == c


class ExecutionHarness:
    """
    Compiles & runs sequential and CUDA candidates.
    IMPORTANT:
      - Pin compiled CUDA executions to GPU0 (rank_gpu=0) so training isn't disturbed.
    """
    def __init__(self,
                 rank_gpu: Optional[int] = 0,
                 gxx="g++", nvcc="nvcc",
                 cflags="-O3 -march=native",
                 nvccflags="-O3",
                 trials_per_input=TRIALS_PER_INPUT,
                 warmup_runs=WARMUP_RUNS,
                 run_timeout_sec=RUN_TIMEOUT_SEC):
        self.rank_gpu = rank_gpu
        self.gxx = gxx
        self.nvcc = nvcc
        self.cflags = shlex.split(cflags)
        self.nvccflags = shlex.split(nvccflags)
        self.trials_per_input = trials_per_input
        self.warmup_runs = warmup_runs
        self.run_timeout_sec = run_timeout_sec

    def _env(self) -> Dict[str, str]:
        if self.rank_gpu is None:
            return {}
        return {"CUDA_VISIBLE_DEVICES": str(self.rank_gpu)}

    def _compile_c(self, src_path: str, out_path: str) -> Tuple[bool, str]:
        cmd = [self.gxx, src_path, "-o", out_path, *self.cflags]
        rc, out, err = _run_cmd(cmd, timeout=self.run_timeout_sec, extra_env=self._env())
        if rc != 0:
            print(f"[HARNESS] C compile FAILED: rc={rc}\nstderr:\n{err}")
        return (rc == 0, err if rc != 0 else "")

    def _compile_cuda(self, cu_path: str, out_path: str) -> Tuple[bool, str]:
        cmd = [self.nvcc, cu_path, "-o", out_path, *self.nvccflags]
        rc, out, err = _run_cmd(cmd, timeout=self.run_timeout_sec, extra_env=self._env())
        if rc != 0:
            print(f"[HARNESS] CUDA compile FAILED: rc={rc}\nstderr:\n{err}")
        return (rc == 0, err if rc != 0 else "")

    def _time_binary(self, exe: str, args: List[str]) -> Tuple[int, str, str, float]:
        stdin_str = " ".join(args) + "\n"
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
            out, err = proc.communicate(stdin_str, timeout=self.run_timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            return -1, "", "TIMEOUT", (time.perf_counter() - start) * 1000.0
        end = time.perf_counter()
        return proc.returncode, out, err, (end - start) * 1000.0

    def _bench(self, exe: str, inputs: List[List[str]]) -> Tuple[List[float], float, List[str]]:
        all_times: List[float] = []
        medians: List[float] = []
        outputs: List[str] = []

        for args in inputs:
            for _ in range(self.warmup_runs):
                rc, out, err, t_ms = self._time_binary(exe, args)
                if rc != 0:
                    return [], 0.0, []

            times_this_input: List[float] = []
            out_this_input: Optional[str] = None

            for _ in range(self.trials_per_input):
                rc, out, err, t_ms = self._time_binary(exe, args)
                if rc != 0:
                    return [], 0.0, []
                times_this_input.append(t_ms)
                all_times.append(t_ms)
                out_this_input = out.strip()

                print(
                    f"[HARNESS] run: exe={exe}, stdin='{ ' '.join(args) }', "
                    f"time={t_ms:.3f} ms, rc={rc}\n[HARNESS] stdout:\n{out}"
                )

            medians.append(_median_ms(times_this_input))
            outputs.append(out_this_input if out_this_input is not None else "")

        cv = _cv(all_times)
        return medians, cv, outputs

    def run(self, full_text: str, task: Task) -> CandMetrics:
        decision_txt = _extract_first(full_text, TAG_RX["DECISION"]).upper()
        decision = (
            "PARALLELIZE"
            if "PARALLEL" in decision_txt
            else "ABSTAIN"
            if "ABSTAIN" in decision_txt
            else "KEEP"
        )
        seq_code = _extract_first(full_text, TAG_RX["SEQ_CODE"])
        cuda_code = _extract_first(full_text, TAG_RX["CUDA_CODE"])
        features = extract_struct_features(cuda_code)
        inputs = _parse_inputs(getattr(task, "inputs_text", "")) or [[]]

        cpu_medians: List[float] = []
        gpu_medians: List[float] = []
        seq_outs: List[str] = []
        cuda_outs: List[str] = []
        compile_ok = True
        compile_errs: List[str] = []

        with tempfile.TemporaryDirectory() as td:
            seq_src = os.path.join(td, "seq.cpp")
            seq_exe = os.path.join(td, "seq.out")
            with open(seq_src, "w") as f:
                f.write(seq_code)
            ok, err = self._compile_c(seq_src, seq_exe)
            compile_ok &= ok
            if not ok:
                compile_errs.append("SEQ COMPILE ERROR:\n" + err[:1000])
                return CandMetrics(
                    False, 0.0, [], [], 0.0, decision, False,
                    seq_outputs=[], cuda_outputs=[], compile_errs=compile_errs,
                )

            has_cuda = bool(cuda_code.strip())

            if decision == "PARALLELIZE" and has_cuda:
                cu_src = os.path.join(td, "kern.cu")
                cu_exe = os.path.join(td, "kern.out")
                with open(cu_src, "w") as f:
                    f.write(cuda_code)

                ok, err = self._compile_cuda(cu_src, cu_exe)
                compile_ok &= ok
                if not ok:
                    compile_errs.append("CUDA COMPILE ERROR:\n" + err[:2000])
                    return CandMetrics(
                        False, 0.0, [], [], 0.0, decision, False,
                        seq_outputs=[], cuda_outputs=[], compile_errs=compile_errs,
                    )

                cpu_medians, cv_cpu, seq_outs = self._bench(seq_exe, inputs)
                gpu_medians, cv_gpu, cuda_outs = self._bench(cu_exe, inputs)
                cv = max(cv_cpu, cv_gpu)

                for i, args in enumerate(inputs):
                    so = seq_outs[i] if i < len(seq_outs) else ""
                    co = cuda_outs[i] if i < len(cuda_outs) else ""
                    equal = _outputs_match(so, co)
                    print(
                        "[HARNESS] SUMMARY input={} | "
                        "seq_ms={:.3f}, cuda_ms={:.3f}, "
                        "seq_out={}, cuda_out={}, equal={}".format(
                            " ".join(args),
                            cpu_medians[i] if i < len(cpu_medians) else float("nan"),
                            gpu_medians[i] if i < len(gpu_medians) else float("nan"),
                            so, co, equal,
                        )
                    )

                pass_rate = _compute_pass_rate(seq_outs, cuda_outs)
                valid = bool(cpu_medians and gpu_medians and pass_rate > 0.0)

                return CandMetrics(
                    compile_ok, pass_rate, cpu_medians, gpu_medians, cv, decision, valid,
                    features=features, seq_outputs=seq_outs, cuda_outputs=cuda_outs, compile_errs=compile_errs,
                )

            # KEEP path: run seq only
            cpu_medians, cv_cpu, seq_outs = self._bench(seq_exe, inputs)
            gpu_medians = cpu_medians[:]
            cuda_outs = seq_outs[:]
            pass_rate = 1.0 if seq_outs else 0.0
            valid = bool(cpu_medians)

            return CandMetrics(
                compile_ok, pass_rate, cpu_medians, gpu_medians, cv_cpu, decision, valid,
                seq_outputs=seq_outs, cuda_outputs=cuda_outs, compile_errs=compile_errs,
            )


# =========================
# 4) REWARD + GRPO
# =========================
def structure_reward(features: Dict[str, float]) -> float:
    x = torch.tensor([features[k] for k in RANKER["feature_names"]], device="cuda:0")
    x = (x - MEAN_f) / STD_f
    return float((x @ W_f).item())

def reward_from_metrics(m: CandMetrics) -> float:
    if not m.compile_ok:
        return -PENAL_FAIL_COMPILE

    correctness_penalty = -PENAL_FAIL_WRONG * (1.0 - m.pass_rate)
    if m.pass_rate <= 0.0:
        return float(correctness_penalty)

    r_struct = structure_reward(m.features) if m.features else 0.0

    if m.decision == "PARALLELIZE" and m.gpu_medians_ms and m.cpu_medians_ms:
        S = max(m.cpu_medians_ms[0] / max(m.gpu_medians_ms[0], 1e-6), 1e-3)
        r_speed = ALPHA_SPEED * float(torch.clamp(torch.tensor(math.log2(S)), *CLIP_LOG2_S))
        return float(
            correctness_penalty
            + r_speed
            + 0.2 * r_struct
            - PENAL_CV * m.cv
        )

    return float(correctness_penalty + BONUS_KEEP_GOOD * m.pass_rate)

def grpo_loss_backward(policy: "Policy", texts: List[str], old_logps: torch.Tensor, rewards: torch.Tensor) -> float:
    total_loss_val = 0.0
    for i, text in enumerate(texts):
        new_lp, _ = policy.masked_logprob(text)
        ratio = torch.exp(new_lp - old_logps[i])
        adv = rewards[i]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
        sample_loss = -torch.min(unclipped, clipped)
        sample_loss.backward()
        total_loss_val += float(sample_loss.detach().item())
    return total_loss_val / max(1, len(texts))


# =========================
# 5) SELF-REFINEMENT
# =========================
def build_feedback(task: Task, metrics: CandMetrics) -> str:
    lines = []
    if not metrics.compile_ok:
        lines.append("Your previous CUDA program failed to compile with nvcc.")
        if metrics.compile_errs:
            lines.append("Here are excerpts of the compiler errors:")
            for err in metrics.compile_errs:
                snippet = err.strip()
                if len(snippet) > 800:
                    snippet = snippet[:800] + "\n… [truncated]"
                lines.append(snippet)
        return "\n".join(lines)

    if metrics.pass_rate < 1.0:
        lines.append(
            f"Your previous CUDA program compiled but only {metrics.pass_rate*100:.1f}% "
            f"of outputs matched the sequential program."
        )
        if metrics.seq_outputs and metrics.cuda_outputs:
            lines.append("Example of a mismatched output (sequential vs CUDA):")
            for s_out, c_out in zip(metrics.seq_outputs, metrics.cuda_outputs):
                if s_out.strip() != c_out.strip():
                    lines.append(f"  sequential: {s_out.strip()}")
                    lines.append(f"  CUDA:       {c_out.strip()}")
                    break
    else:
        lines.append("Your previous CUDA program compiled and produced correct outputs for all tested inputs.")

    if metrics.decision == "PARALLELIZE" and metrics.gpu_medians_ms and metrics.cpu_medians_ms:
        cpu_ms = metrics.cpu_medians_ms[0]
        gpu_ms = metrics.gpu_medians_ms[0]
        S = cpu_ms / max(gpu_ms, 1e-6)
        lines.append(f"Sequential median time: {cpu_ms:.3f} ms, CUDA median time: {gpu_ms:.3f} ms (speedup ≈ {S:.2f}x).")

    return "\n".join(lines)

def generate_population_and_refine(
    policy: "Policy",
    harness: ExecutionHarness,
    task: Task,
    fewshot_pool: Optional[List[FewshotExample]] = None,
    n_init: int = N_INIT_VARIANTS,
    n_refine: int = N_REFINE_STEPS,
) -> Tuple[List[str], List[CandMetrics]]:
    texts: List[str] = []
    metrics_list: List[CandMetrics] = []

    base_user_prompt = task_to_prompt(task, fewshot_pool=fewshot_pool)

    init_texts: List[str] = []
    init_metrics: List[CandMetrics] = []

    for i in range(max(1, n_init)):
        temp = TEMPS[i % len(TEMPS)]
        full_text = policy.generate_candidate(base_user_prompt, temperature=temp, top_p=TOP_P)
        init_texts.append(full_text)
        m = harness.run(full_text, task)
        init_metrics.append(m)

    texts.extend(init_texts)
    metrics_list.extend(init_metrics)

    rewards_init = [reward_from_metrics(m) for m in init_metrics]
    best_idx = max(range(len(rewards_init)), key=lambda idx: rewards_init[idx])
    best_text = init_texts[best_idx]
    best_metric = init_metrics[best_idx]

    if n_refine <= 0:
        return texts, metrics_list

    prev_full_text = best_text
    prev_metric = best_metric

    for r in range(n_refine):
        fb = build_feedback(task, prev_metric)
        prev_cuda = _extract_first(prev_full_text, TAG_RX["CUDA_CODE"])

        prev_cuda_block = (
            "[PREV_CUDA]\n"
            + (prev_cuda if prev_cuda else "// (Previous CUDA attempt was empty or could not be extracted)\n")
            + "\n[/PREV_CUDA]\n"
        )

        refined_user_prompt = (
            base_user_prompt
            + "\n\n"
            + prev_cuda_block
            + "The CUDA code above is your previous attempt and has the following issues:\n"
            + fb
            + "\n\nPlease generate a NEW corrected CUDA C++ program that:\n"
              "  - matches the behavior of the original sequential C++ program for all inputs, and\n"
              "  - compiles successfully with nvcc as a single translation unit.\n"
              "Only output CUDA C++ code (no explanations, no markdown, no comments about what you are doing).\n"
        )
        if USE_NO_THINK:
            refined_user_prompt += "\n/no_think\n"

        temp = TEMPS[(n_init + r) % len(TEMPS)]
        full_text = policy.generate_candidate(refined_user_prompt, temperature=temp, top_p=TOP_P)
        texts.append(full_text)

        m = harness.run(full_text, task)
        metrics_list.append(m)

        prev_full_text = full_text
        prev_metric = m

        if m.compile_ok and m.pass_rate >= 1.0:
            break

    return texts, metrics_list


# =========================
# 6) TRAIN STEP
# =========================
def rlaif_step_single_process(
    policy: "Policy",
    harness: ExecutionHarness,
    task: Task,
    fewshot_pool: Optional[List[FewshotExample]] = None
) -> Dict:
    texts, metrics = generate_population_and_refine(
        policy=policy,
        harness=harness,
        task=task,
        fewshot_pool=fewshot_pool,
        n_init=N_INIT_VARIANTS,
        n_refine=N_REFINE_STEPS
    )

    with torch.no_grad():
        old_lps = torch.stack([policy.masked_logprob(t)[0] for t in texts]).to("cuda:0")

    rewards = torch.tensor([reward_from_metrics(m) for m in metrics], device="cuda:0", dtype=torch.float32)
    loss_grpo_val = grpo_loss_backward(policy, texts, old_lps, rewards)

    return {
        "loss_grpo": float(loss_grpo_val),
        "rewards_mean": float(rewards.mean().item()),
        "best_reward": float(rewards.max().item()),
    }


# =========================
# 7) BUILD MODEL (Qwen3-8B)
# =========================
def _select_lora_targets(model, wanted: List[str]) -> List[str]:
    present = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        present.add(last)
    keep = [w for w in wanted if w in present]
    if not keep:
        for w in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if w in present:
                keep.append(w)
    return keep

def _infer_max_memory() -> Optional[Dict[int, str]]:
    """
    For 8B, you can usually fit on a single 40GB A100 easily in 4-bit + LoRA.
    If 2 GPUs are visible, we still provide a conservative split.
    """
    n = torch.cuda.device_count()
    if n <= 0:
        return None
    if n == 1:
        return {0: "38GiB"}
    return {0: "38GiB", 1: "38GiB"}

def build_policy() -> Policy:
    tok = os.environ.get("HUGGINGFACE_TOKEN", "")
    if tok:
        login(token=tok)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.add_special_tokens(SPECIALS)

    max_mem = _infer_max_memory()

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

    wanted = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    targets = _select_lora_targets(model, wanted)
    print(f"[INFO] LoRA target_modules = {targets}")

    # For 8B you can often use a bit higher rank; keep same unless you want to change.
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


def get_ground_truth_speedup(task: Task) -> Optional[float]:
    meta = None
    if getattr(task, "extra", None):
        meta = task.extra.get("meta") or task.extra.get("extra") or task.extra
    if not isinstance(meta, dict):
        return None

    CAND_KEYS = [
        "gt_speedup",
        "ground_truth_speedup",
        "speedup",
        "avg_speedup",
        "best_speedup",
        "cuda_speedup",
        "oracle_speedup",
        "speedup_gt",
        "max_speedup",
    ]

    for k in CAND_KEYS:
        if k in meta:
            v = meta.get(k)
            try:
                fv = float(v)
                if math.isfinite(fv) and fv > 0.0:
                    return fv
            except Exception:
                pass

    for nest in ["metrics", "perf", "performance", "eval"]:
        sub = meta.get(nest)
        if isinstance(sub, dict):
            for k in CAND_KEYS:
                if k in sub:
                    v = sub.get(k)
                    try:
                        fv = float(v)
                        if math.isfinite(fv) and fv > 0.0:
                            return fv
                    except Exception:
                        pass

    return None


@torch.no_grad()
def final_evaluate(policy: Policy, harness: ExecutionHarness, tasks: List[Task], fewshot_pool) -> None:
    print("\n====================")
    print("Running final evaluation on all tasks...")
    print("====================\n")

    policy.eval()

    total_tasks_compiled = 0
    fully_correct_cases = 0
    speedup_cases = 0
    speedup_gt1_cases = 0
    total_speedup = 0.0

    max_possible_speedup = 0.0
    possible_speedup_cases = 0
    sum_possible_speedup = 0.0
    gt_and_model_speedup_cases = 0

    eval_temp = 0.2
    eval_top_p = TOP_P

    for task in tasks:
        gt_speed = get_ground_truth_speedup(task)
        if gt_speed is not None:
            if gt_speed > max_possible_speedup:
                max_possible_speedup = gt_speed
            if gt_speed > 1.0:
                possible_speedup_cases += 1
                sum_possible_speedup += gt_speed

        user_prompt = task_to_prompt(task, fewshot_pool=fewshot_pool)
        full_text = policy.generate_candidate(user_prompt, temperature=eval_temp, top_p=eval_top_p)
        metrics = harness.run(full_text, task)

        if not metrics.compile_ok:
            continue

        total_tasks_compiled += 1

        if metrics.pass_rate >= 1.0:
            fully_correct_cases += 1

        if (
            metrics.decision == "PARALLELIZE"
            and metrics.gpu_medians_ms
            and metrics.cpu_medians_ms
            and metrics.pass_rate > 0.0
        ):
            seq_t = metrics.cpu_medians_ms[0]
            gpu_t = metrics.gpu_medians_ms[0]
            speed = seq_t / max(gpu_t, 1e-6)

            total_speedup += speed
            speedup_cases += 1

            if speed > 1.0:
                speedup_gt1_cases += 1
                if gt_speed is not None and gt_speed > 1.0:
                    gt_and_model_speedup_cases += 1

    avg_speedup = total_speedup / speedup_cases if speedup_cases > 0 else 0.0
    avg_possible_speedup = (sum_possible_speedup / possible_speedup_cases) if possible_speedup_cases > 0 else 0.0

    print("\n==================== FINAL EVAL REPORT ====================")
    print(f"Model                                              : {MODEL_NAME}")
    print(f"Total tasks that compiled in eval                 : {total_tasks_compiled}")
    print(f"Tasks with all testcases matched                  : {fully_correct_cases}")
    print(f"Tasks with measurable CUDA speedup (any)          : {speedup_cases}")
    print(f"Tasks with model-achieved speedup > 1             : {speedup_gt1_cases}")
    print(f"Average model-achieved speedup over measured tasks: {avg_speedup:.3f}")
    print()
    print(f"Ground-truth: tasks where speedup is possible (GT>1): {possible_speedup_cases}")
    print(f"Ground-truth: max possible speedup                  : {max_possible_speedup:.3f}")
    print(f"Ground-truth: avg possible speedup over GT>1 tasks  : {avg_possible_speedup:.3f}")
    denom = possible_speedup_cases if possible_speedup_cases > 0 else 1
    print(
        f"Tasks where GT says speedup is possible AND "
        f"model achieved >1x speedup: {gt_and_model_speedup_cases}/{denom}"
    )
    print("===========================================================\n")

    policy.train()


def export_task_result_json(
    out_dir: str,
    task: Task,
    best_full_text: str,
    best_metrics: CandMetrics,
    reward: float,
    extra: Optional[Dict] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    seq_code = _extract_first(best_full_text, TAG_RX["SEQ_CODE"])
    cuda_code = _extract_first(best_full_text, TAG_RX["CUDA_CODE"])

    sp = compute_speedups(best_metrics.cpu_medians_ms, best_metrics.gpu_medians_ms)

    payload = {
        "task_id": task.task_id,
        "decision": best_metrics.decision,
        "reward": _safe_float(reward),
        "compile_ok": bool(best_metrics.compile_ok),
        "valid": bool(best_metrics.valid),
        "pass_rate": _safe_float(best_metrics.pass_rate),
        "cv": _safe_float(best_metrics.cv),

        "cpu_medians_ms": [_safe_float(x) for x in (best_metrics.cpu_medians_ms or [])],
        "gpu_medians_ms": [_safe_float(x) for x in (best_metrics.gpu_medians_ms or [])],

        **sp,

        "seq_code": seq_code,
        "cuda_code": cuda_code,

        "features": best_metrics.features or {},
        "compile_errs": best_metrics.compile_errs or [],
    }

    if extra:
        payload["extra"] = extra

    out_path = os.path.join(out_dir, f"{task.task_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[EXPORT] Wrote {out_path}")


@torch.no_grad()
def optimize_and_export_all_tasks(
    policy: Policy,
    harness: ExecutionHarness,
    tasks: List[Task],
    fewshot_pool: Optional[List[FewshotExample]],
    out_dir: str,
):
    print("\n====================")
    print("Optimizing + exporting per-task best CUDA programs...")
    print("====================\n")

    policy.eval()

    for task in tasks:
        texts, metrics_list = generate_population_and_refine(
            policy=policy,
            harness=harness,
            task=task,
            fewshot_pool=fewshot_pool,
            n_init=N_INIT_VARIANTS,
            n_refine=N_REFINE_STEPS,
        )

        rewards = [reward_from_metrics(m) for m in metrics_list]
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])

        best_text = texts[best_idx]
        best_metrics = metrics_list[best_idx]
        best_reward = rewards[best_idx]

        gt_speed = get_ground_truth_speedup(task)
        extra = {"gt_speedup": gt_speed} if gt_speed is not None else {}

        export_task_result_json(
            out_dir=out_dir,
            task=task,
            best_full_text=best_text,
            best_metrics=best_metrics,
            reward=best_reward,
            extra=extra,
        )

    policy.train()


def save_policy(policy: Policy, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    print(f"[SAVE] Saving LoRA adapters to {save_dir}")
    policy.model.save_pretrained(save_dir)
    if SAVE_TOKENIZER:
        print(f"[SAVE] Saving tokenizer to {save_dir}")
        policy.tokenizer.save_pretrained(save_dir)


# =========================
# 9) MAIN
# =========================
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA devices visible to PyTorch.")

    policy = build_policy()
    policy.train()

    if HAVE_BNB:
        optimizer = bnb.optim.PagedAdamW8bit(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, TOTAL_STEPS)

    # Harness pinned to GPU0 for compiled CUDA runs
    harness = ExecutionHarness(rank_gpu=0)

    tasks, seeds, fewshot_pool = load_tasks_and_seeds_from_jsonl(JSONL_PATH, drop_bad_parallel=True)
    print(f"[INFO] Loaded tasks={len(tasks)} seeds={len(seeds)} fewshot_pool={len(fewshot_pool)}")

    random.seed(1337)
    torch.manual_seed(1337)

    num_epochs = 1
    print_every = 1
    step = 0

    for epoch in range(num_epochs):
        random.shuffle(tasks)
        for task in tasks:
            optimizer.zero_grad(set_to_none=True)

            out = rlaif_step_single_process(policy, harness, task, fewshot_pool=fewshot_pool)

            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % print_every == 0:
                print(
                    f"[{step}] grpo_loss={out['loss_grpo']:.3f} | "
                    f"R̄={out['rewards_mean']:.3f} bestR={out['best_reward']:.3f} | "
                    f"task={task.task_id}"
                )
            step += 1

    print("Training finished.")

    final_evaluate(policy, harness, tasks, fewshot_pool)

    optimize_and_export_all_tasks(
        policy=policy,
        harness=harness,
        tasks=tasks,
        fewshot_pool=fewshot_pool,
        out_dir=EXPORT_DIR
    )

    save_policy(policy, SAVE_DIR)
    print("[SAVE] Done.")


if __name__ == "__main__":
    main()
