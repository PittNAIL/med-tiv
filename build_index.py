#!/usr/bin/env python3
"""
build_index.py - FIXED: Increased Queue sizes to prevent deadlocks
"""

import argparse
import gc
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import faiss
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# -----------------------------
# SAME pooling logic as retrieval_server.py
# -----------------------------
def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def _maybe_add_prefix(texts: List[str], is_query: bool, model_name: str) -> List[str]:
    """
    Add model-specific prefixes for asymmetric encoders.
    
    For E5: Uses "query:" and "passage:" prefixes
    For MedCPT: No prefixes needed (uses separate models)
    """
    if "e5" in model_name.lower():
        if is_query:
            return [f"query: {t}" for t in texts]
        else:
            return [f"passage: {t}" for t in texts]
    elif "medcpt" in model_name.lower():
        # MedCPT uses separate Article/Query encoders
        # No text prefixes needed
        return texts
    return texts


@torch.no_grad()
def encode_batch_transformers(
    model,
    tokenizer,
    texts: List[str],
    model_name: str,
    pooling_method: str,
    max_length: int,
) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if "T5" in type(model).__name__:
        decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long, device=device)
        output = model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
        emb = output.last_hidden_state[:, 0, :]
    else:
        output = model(**inputs, return_dict=True)
        emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], pooling_method)
        if "dpr" not in model_name.lower():
            emb = torch.nn.functional.normalize(emb, dim=-1)

    emb = emb.detach().cpu().numpy().astype(np.float32, order="C")

    del inputs, output
    return emb


def worker_loop(
    gpu_id: int,
    model_path: str,
    model_name: str,
    pooling_method: str,
    max_length: int,
    use_fp16: bool,
    in_q: mp.Queue,
    out_q: mp.Queue,
):
    try:
        torch.cuda.set_device(gpu_id)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().cuda()
        if use_fp16:
            model = model.half()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

        while True:
            item = in_q.get()
            if item is None:
                break
            batch_id, texts = item
            embs = encode_batch_transformers(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                model_name=model_name,
                pooling_method=pooling_method,
                max_length=max_length,
            )
            out_q.put((batch_id, embs))

    except Exception as e:
        out_q.put(("__error__", f"GPU {gpu_id} worker failed: {repr(e)}"))
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def build_index(
    corpus_path: str,
    index_output: str,
    model_path: str,
    model_name: str,
    pooling_method: str = "mean",
    max_length: int = 256,
    batch_size: int = 512,
    use_fp16: bool = False,
    gpus: Optional[List[int]] = None,
    max_docs: Optional[int] = None,
    meta: bool = True,
):
    corpus_path = str(corpus_path)
    index_output = str(index_output)
    out_path = Path(index_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if gpus is None:
        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            raise RuntimeError("No CUDA GPUs detected.")
        gpus = list(range(gpu_count))
    if len(gpus) < 1:
        raise RuntimeError("At least one GPU is required.")

    # Get embedding dim
    torch.cuda.set_device(gpus[0])
    tmp_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().cuda()
    if use_fp16:
        tmp_model = tmp_model.half()
    tmp_tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    probe = _maybe_add_prefix(["test"], is_query=False, model_name=model_name)
    probe_emb = encode_batch_transformers(tmp_model, tmp_tok, probe, model_name, pooling_method, max_length)
    emb_dim = int(probe_emb.shape[1])
    del tmp_model, tmp_tok, probe_emb
    torch.cuda.empty_cache()

    index = faiss.IndexFlatIP(emb_dim)

    with open(corpus_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    if max_docs:
        total_lines = min(total_lines, max_docs)

    print("=" * 70)
    print("Building FAISS Index (Fixed Deadlock Version)")
    print("=" * 70)
    print(f"Corpus: {corpus_path}")
    print(f"Index output: {index_output}")
    print(f"Model path: {model_path}")
    print(f"GPUs: {gpus}")
    print("=" * 70)

    # ---------------------------------------------------------
    # FIX: INCREASED QUEUE SIZES TO PREVENT DEADLOCK
    # ---------------------------------------------------------
    ctx = mp.get_context("spawn")
    # Buffer enough batches so main process doesn't block frequently
    in_queues = [ctx.Queue(maxsize=128) for _ in gpus]
    # Huge output buffer to ensure workers NEVER block on put()
    out_q = ctx.Queue(maxsize=1024)
    
    procs = []
    for i, gpu_id in enumerate(gpus):
        p = ctx.Process(
            target=worker_loop,
            args=(gpu_id, model_path, model_name, pooling_method, max_length, use_fp16, in_queues[i], out_q),
            daemon=True,
        )
        p.start()
        procs.append(p)

    next_batch_id_to_add = 0
    pending: Dict[int, np.ndarray] = {}

    batch_id = 0
    total_indexed = 0
    rr = 0
    batch_texts: List[str] = []

    def _finalize_text(line: str) -> str:
        if not line.strip():
            text = "empty_document_placeholder"
            return _maybe_add_prefix([text], is_query=False, model_name=model_name)[0]
        try:
            doc = json.loads(line)
            text = doc.get("text", "")
        except json.JSONDecodeError:
            text = "json_error_placeholder"
            return _maybe_add_prefix([text], is_query=False, model_name=model_name)[0]

        if not text or len(text.strip()) < 10:
            title = doc.get("title", "")
            if isinstance(title, str) and len(title.strip()) > 5:
                text = title.strip()
            else:
                text = "empty_document_placeholder"

        text = _maybe_add_prefix([text], is_query=False, model_name=model_name)[0]
        return text

    def _dispatch(batch_id: int, texts: List[str]):
        nonlocal rr
        q = in_queues[rr]
        rr = (rr + 1) % len(in_queues)
        q.put((batch_id, texts))

    def _drain_and_add(pbar):
        nonlocal next_batch_id_to_add, total_indexed, pending
        # Aggressively drain output queue
        while True:
            try:
                item = out_q.get_nowait()
            except Exception:
                break

            if isinstance(item, tuple) and item and item[0] == "__error__":
                raise RuntimeError(item[1])

            bid, embs = item
            pending[bid] = embs

        while next_batch_id_to_add in pending:
            embs = pending.pop(next_batch_id_to_add)
            index.add(embs)
            n = embs.shape[0]
            total_indexed += n
            pbar.update(n)
            next_batch_id_to_add += 1
            if next_batch_id_to_add % 50 == 0:
                gc.collect()

    with open(corpus_path, "r", encoding="utf-8") as f, tqdm(total=total_lines, desc="Indexing", unit="doc") as pbar:
        for line_idx, line in enumerate(f):
            if max_docs and (line_idx >= max_docs):
                break

            batch_texts.append(_finalize_text(line))

            if len(batch_texts) >= batch_size:
                _dispatch(batch_id, batch_texts)
                batch_id += 1
                batch_texts = []
                # Drain frequently to keep out_q empty
                _drain_and_add(pbar)

        if batch_texts:
            _dispatch(batch_id, batch_texts)
            batch_id += 1
            batch_texts = []

        # Final wait
        while next_batch_id_to_add < batch_id:
            item = out_q.get()
            if isinstance(item, tuple) and item and item[0] == "__error__":
                raise RuntimeError(item[1])
            bid, embs = item
            pending[bid] = embs
            _drain_and_add(pbar)

    for q in in_queues:
        q.put(None)
    for p in procs:
        p.join(timeout=30)

    print(f"\n✓ Indexed {total_indexed:,} vectors")
    print(f"Saving index to: {index_output}")
    faiss.write_index(index, index_output)

    if meta:
        meta_path = str(out_path) + ".meta.json"
        meta_obj = {
            "corpus_path": str(Path(corpus_path).resolve()),
            "index_path": str(Path(index_output).resolve()),
            "model_path": str(Path(model_path).resolve()),
            "model_name_tag": model_name,
            "embedding_dim": emb_dim,
            "faiss_index_type": "IndexFlatIP",
            "pooling_method": pooling_method,
            "max_length": max_length,
            "normalized": ("dpr" not in model_name.lower()),
            "e5_passage_prefix": ("e5" in model_name.lower()),
            "total_indexed": total_indexed,
        }
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta_obj, mf, ensure_ascii=False, indent=2)

    print("✓ Done!")

def parse_gpus(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--index_output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_name_tag", default="e5")
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls", "pooler"])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--no_meta", action="store_true")

    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    gpus = parse_gpus(args.gpus) if args.gpus else None

    build_index(
        corpus_path=args.corpus_path,
        index_output=args.index_output,
        model_path=args.model,
        model_name=args.model_name_tag,
        pooling_method=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        gpus=gpus,
        max_docs=args.max_docs,
        meta=(not args.no_meta),
    )