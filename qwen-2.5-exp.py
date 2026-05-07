#!/usr/bin/env python3
# Works for: 2× A100 40GB with QLoRA (4-bit) + LoRA on a 32B Qwen Coder model
#
# Install (example):
#   pip install -U transformers==4.42.0 peft accelerate bitsandbytes huggingface_hub torch numpy
#
# Run:
#   export CUDA_VISIBLE_DEVICES=0,1
#   export HUGGINGFACE_TOKEN=...
#   python train_qwen32b_qlora_grpo.py
#
# Notes:
# - Single-process training (NO DDP). Model shards across both GPUs using device_map="auto".
# - Harness runs compiled CUDA programs pinned to GPU0 to avoid contention.
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



SAVE_DIR = "checkpoints/qwen32b-qlora-grpo-full-dataset-2"
SAVE_TOKENIZER = True

# =========================
# 0) CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-32B"

# Training mode for 2×A100 40GB
USE_4BIT_QLORA = True
TRAIN_DTYPE = torch.bfloat16

SYSTEM_PROMPT = "You are an expert CUDA C++ engineer."

# Exploration + refinement
N_INIT_VARIANTS = 4
N_REFINE_STEPS = 3

TEMPS = [0.2, 0.7]
TOP_P = 0.9

# IMPORTANT: keep this sane for 32B
MAX_STEP_TOK = 1536

# Masked logprob cap to avoid OOM
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

# Optim
LR = 1e-5
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
FEWSHOT_K = 2                 # how many examples to include per prompt
FEWSHOT_MIN_SPEEDUP = 1.01    # threshold
FEWSHOT_MAX_CHARS_PER_CODE = 8000  # safety cap to avoid exploding prompt
FEWSHOT_SEED = 1337

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

        # seeds (as before)
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

        # few-shot pool: only if speedup > 1 and has CUDA
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
    """
    Returns a string block to prepend to the user prompt.
    Samples k examples from pool excluding current_task_id.
    """
    if not USE_FEWSHOT_CONTEXT or k <= 0 or not pool:
        return ""

    # filter out current task
    cand = [ex for ex in pool if ex.task_id != current_task_id]
    if not cand:
        return ""

    # bias toward higher speedups
    cand.sort(key=lambda ex: ex.speedup, reverse=True)

    # sample from top-N to keep quality + diversity
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
        blocks.append("")  # spacing

    blocks.append("[/FEWSHOT]")
    blocks.append("")  # final spacing

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
    Single-process wrapper around a sharded (device_map="auto") causal LM.
    IMPORTANT:
      - Do NOT .to(device) the model; it is already sharded.
      - Put small input tensors on cuda:0; HF dispatches internally.
    """
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tid = {
            t: tokenizer.convert_tokens_to_ids(t)
            for t in SPECIALS["additional_special_tokens"]
        }

        # memory savers
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

    @torch.no_grad()
    def generate_candidate(self, user_prompt: str, temperature: float, top_p: float) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        tok = self.tokenizer(chat_prompt, return_tensors="pt")
        tok.pop("token_type_ids", None)
        tok = {k: v.to("cuda:0") for k, v in tok.items()}  # tiny tensors
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

        # Heuristic: drop any leading explanation; start from first sign of code
        start_idx = len(raw)
        for marker in ["#include", "__global__", "int main", "using namespace std"]:
            pos = raw.find(marker)
            if pos != -1 and pos < start_idx:
                start_idx = pos
        code = raw[start_idx:] if start_idx < len(raw) else raw

        # Cut after last closing brace
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
        logits = self.model(**tok).logits[0]  # [T, V]
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

def _parse_first_number(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    first = s.split()[0]
    try:
        return float(first)
    except ValueError:
        return None

def _compute_pass_rate(seq_outputs: List[str], cuda_outputs: List[str],
                       atol: float = 1e-6, rtol: float = 1e-6) -> float:
    if not seq_outputs or not cuda_outputs:
        return 0.0
    total = min(len(seq_outputs), len(cuda_outputs))
    if total == 0:
        return 0.0
    matches = 0
    for s_out, c_out in zip(seq_outputs, cuda_outputs):
        s_val = _parse_first_number(s_out)
        c_val = _parse_first_number(c_out)
        if s_val is not None and c_val is not None:
            if abs(s_val - c_val) <= atol + rtol * max(abs(s_val), 1.0):
                matches += 1
        else:
            if s_out.strip() == c_out.strip():
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
    IMPORTANT for 2×A100 training:
      - Pin compiled CUDA executions to GPU0 (rank_gpu=0) so training sharding isn't disturbed.
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
                    seq_outputs=seq_outs, cuda_outputs=cuda_outs, compile_errs=compile_errs,
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
def reward_from_metrics(m: CandMetrics) -> float:
    if not m.compile_ok:
        return -PENAL_FAIL_COMPILE

    correctness_penalty = -PENAL_FAIL_WRONG * (1.0 - m.pass_rate)
    if m.pass_rate <= 0.0:
        return float(correctness_penalty)

    if m.decision == "PARALLELIZE" and m.gpu_medians_ms and m.cpu_medians_ms:
        S = max(m.cpu_medians_ms[0] / max(m.gpu_medians_ms[0], 1e-6), 1e-3)
        r_speed = ALPHA_SPEED * float(
            torch.clamp(torch.tensor(math.log2(S)), *CLIP_LOG2_S)
        )
        r = correctness_penalty + r_speed - PENAL_CV * m.cv
        return float(r)

    return float(correctness_penalty + BONUS_KEEP_GOOD * m.pass_rate)

def grpo_loss_backward(policy: Policy, texts: List[str], old_logps: torch.Tensor, rewards: torch.Tensor) -> float:
    """
    Backward per-sample to reduce memory.
    Returns mean scalar loss (python float) for logging.
    """
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
    policy: Policy,
    harness: ExecutionHarness,
    task: Task,
    fewshot_pool: Optional[List[FewshotExample]] = None,
    n_init: int = N_INIT_VARIANTS,
    n_refine: int = N_REFINE_STEPS,
) -> Tuple[List[str], List[CandMetrics]]:
    texts: List[str] = []
    metrics_list: List[CandMetrics] = []

    base_user_prompt = task_to_prompt(task, fewshot_pool=fewshot_pool)

    # Phase 1: initial population
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

    # Phase 2: refinement
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
def rlaif_step_single_process(policy: Policy, harness: ExecutionHarness, task: Task, fewshot_pool: Optional[List[FewshotExample]] = None) -> Dict:
    
    texts, metrics = generate_population_and_refine(
        policy=policy,
        harness=harness,
        task=task,
        fewshot_pool=fewshot_pool,
        n_init=N_INIT_VARIANTS,
        n_refine=N_REFINE_STEPS
    )


    # frozen old logprobs
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
# 7) BUILD MODEL (2×A100-40)
# =========================
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

    # LoRA adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return Policy(model, tokenizer)

def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n// ... [TRUNCATED]\n"

def _meta_speedup_from_rec(meta: Dict) -> Optional[float]:
    """Parse speedup from a meta dict (same logic as get_ground_truth_speedup but for raw rec/meta)."""
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

def get_ground_truth_speedup(task: Task) -> Optional[float]:
    """
    Tries to read a "ground-truth" speedup from task.extra/meta.
    This depends on what you stored in your JSONL `meta`.
    We try a bunch of common key names.

    Returns:
      float speedup if found and parseable, else None
    """
    meta = None
    if getattr(task, "extra", None):
        meta = task.extra.get("meta") or task.extra.get("extra") or task.extra
    if not isinstance(meta, dict):
        return None

    # common candidates in datasets
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

    # Sometimes nested, e.g. meta["metrics"]["avg_speedup"]
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

    # Model-achieved metrics
    total_tasks_compiled = 0
    fully_correct_cases = 0
    speedup_cases = 0            # tasks where we measured CUDA speedup (any speed)
    speedup_gt1_cases = 0        # tasks where model achieved >1x speedup
    total_speedup = 0.0          # sum of achieved speedups over measured tasks

    # Ground-truth metrics
    max_possible_speedup = 0.0
    possible_speedup_cases = 0
    sum_possible_speedup = 0.0
    gt_and_model_speedup_cases = 0

    # Evaluation generation settings (deterministic-ish)
    eval_temp = 0.2
    eval_top_p = TOP_P

    for task in tasks:
        # --- ground-truth speedup from metadata (if present) ---
        gt_speed = get_ground_truth_speedup(task)
        if gt_speed is not None:
            if gt_speed > max_possible_speedup:
                max_possible_speedup = gt_speed
            if gt_speed > 1.0:
                possible_speedup_cases += 1
                sum_possible_speedup += gt_speed

        # --- model evaluation ---
        user_prompt = task_to_prompt(task, fewshot_pool=fewshot_pool)

        # single sample
        full_text = policy.generate_candidate(
            user_prompt,
            temperature=eval_temp,
            top_p=eval_top_p,
        )

        metrics = harness.run(full_text, task)

        if not metrics.compile_ok:
            continue

        total_tasks_compiled += 1

        if metrics.pass_rate >= 1.0:
            fully_correct_cases += 1

        # Only consider speed when we have a CUDA/parallel run and some correctness
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
    avg_possible_speedup = (
        sum_possible_speedup / possible_speedup_cases
        if possible_speedup_cases > 0
        else 0.0
    )

    print("\n==================== FINAL EVAL REPORT ====================")
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


# =============================
# 8) Save Model for inference
# =============================

def save_policy(policy: Policy, save_dir: str):
    """
    Saves LoRA adapters + tokenizer only.
    Safe for QLoRA + device_map="auto".
    """
    os.makedirs(save_dir, exist_ok=True)

    # policy.model is a PEFT model
    print(f"[SAVE] Saving LoRA adapters to {save_dir}")
    policy.model.save_pretrained(save_dir)

    if SAVE_TOKENIZER:
        print(f"[SAVE] Saving tokenizer to {save_dir}")
        policy.tokenizer.save_pretrained(save_dir)

# =========================
# 8) MAIN
# =========================
def main():
    # perf knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if torch.cuda.device_count() < 2:
        print(f"[WARN] torch sees only {torch.cuda.device_count()} CUDA device(s). "
              f"Expected 2 for 2×A100 sharding. Continuing anyway.")

    policy = build_policy()
    policy.train()

    # Optimizer
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

    # ---- final evaluation ----
    final_evaluate(policy, harness, tasks, fewshot_pool)

    # ---- Save model ----
    save_policy(policy, SAVE_DIR)
    print("[SAVE] Done.")

if __name__ == "__main__":
    main()



# Loading models later

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel

# BASE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
# LORA_DIR = "checkpoints/qwen32b-qlora-grpo"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     quantization_config=bnb_config,
# )

# model = PeftModel.from_pretrained(base_model, LORA_DIR)
# model.eval()
# model.generate(...)
