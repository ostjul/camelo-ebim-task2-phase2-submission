#!/usr/bin/env python3
"""Decode what pi0.5's PaliGemma prefix would GENERATE, across a full episode.

    python scripts/pi05_decode_prefix.py \
        --checkpoint lerobot/pi05_base \
        --dataset outputs/datasets/ext_fixpos200_model16_v1 \
        --episode 0 --stride 30 --max-new-tokens 24

Why this exists. lerobot's pi0.5 computes ONE loss — flow-matching MSE
(`modeling_pi05.py:611`) — and throws the PaliGemma prefix output away
(`:589`, `(_, suffix_out), _ = ...`). The `lm_head` weights are present in
the checkpoint and loaded, but never called. So the model *can* be asked
what it would say; nothing in the training or inference path ever asks.

This probe asks. It is a READ-ONLY observation of an existing checkpoint —
it trains nothing and changes no contract.

## What is and is not faithful here

FAITHFUL: the prompt, the images, the state discretization and the prefix
attention are all built by the checkpoint's own processor pipeline and the
model's own `embed_prefix`, so the hidden states this reads are exactly the
ones the action expert conditions on.

NOT FAITHFUL, and it cannot be: pi0.5's prefix is a single BIDIRECTIONAL
block (`embed_prefix:514` appends `[0] * num_lang_embs`, and 0 means "same
attention block" — `vla_utils.py:61-90`). Nothing in this checkpoint was
ever trained to emit a token at the prefix's last position under lerobot's
objective. Continuation tokens are therefore appended as a NEW causal block
(att_mask 1), which is the only well-posed way to decode, but it is a
prompt format the lerobot fine-tune never saw. Read the output as evidence
about the INHERITED PaliGemma/openpi weights, not as "the subtask lerobot
predicts" — lerobot predicts none.

The right-padding is stripped before decoding. The processor pads to
`tokenizer_max_length` (200) with `padding_side="right"`, so the last
position of the padded prompt is a PAD token, not the last real token;
reading logits there would decode noise and look like a finding.

## Reading the result

The question is whether the decoded string VARIES across an episode, and
if so, what drives the variation. `--baseline` re-decodes with the cameras
set to -1 (SigLIP's pad value); if blanking collapses the output to a
constant, the variation is driven by the IMAGES rather than by the prompt.

Two controls that must be read before the decodes:

- `selfcheck` is teacher-forced next-token accuracy on the prompt's own
  tokens. It separates "the head is dead" from "the head works and the
  answer is genuinely boring". It is reported PER TEMPLATE because a
  prompt carrying the `State: <digits>` block scores near zero on it --
  those ~130 digit tokens are unpredictable as text. That is a property
  of the prompt, not a broken head.
- the blank-image control above.

## What this measured on 2026-08-18 (job 3853573, pi05_base, ep 0 and 7)

- `"{task}. Subtask: "` (openpi issue #701 format, NO state digits) decodes
  fluent, varying, image-driven English: "pick up blue package", "return to
  home position", "hang the towel on oven handle". selfcheck 0.434.
- BOTH templates containing the `State: <digits>` block decode Unicode
  garbage, selfcheck 0.05-0.09 -- including lerobot's REAL action prompt.
- Blanking the cameras collapses ep0 to the single constant string
  "pick up the blue pillow" on all 31 frames.

So the subtask head is alive and image-grounded, but only reachable with a
prompt the action expert never sees. See docs/research/PI05_STATE_ACTION_SUBTASKS.md
section 4.4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="lerobot/pi05_base")
    p.add_argument("--dataset", default="outputs/datasets/ext_fixpos200_model16_v1")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=30, help="sample every Nth frame of the episode")
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--topk", type=int, default=5, help="top-k at the FIRST generated position")
    p.add_argument(
        "--template",
        action="append",
        default=None,
        help="prompt template with {task} and {state} placeholders (repeatable). "
        "Defaults cover lerobot's real action prompt, the same prompt with a "
        "Subtask tail, and the openpi issue #701 format (no state digits).",
    )
    p.add_argument("--baseline", action="store_true", help="also decode with images blanked / state zeroed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="write JSON here")
    p.add_argument("--dry-run", action="store_true", help="check data plumbing without loading the model")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.dataset, root=args.dataset)
    # lerobot 0.6 moved episode bounds onto meta.episodes (there is no
    # `episode_data_index` any more).
    episodes = ds.meta.episodes
    ep_from = int(episodes["dataset_from_index"][args.episode])
    ep_to = int(episodes["dataset_to_index"][args.episode])
    idxs = list(range(ep_from, ep_to, args.stride))[: args.max_frames]
    print(f"episode {args.episode}: frames {ep_from}..{ep_to} ({ep_to - ep_from}), probing {len(idxs)}")

    if args.dry_run:
        s = ds[idxs[0]]
        print("sample keys:", sorted(k for k in s if not k.startswith("observation.images")))
        print("cameras:", sorted(k for k in s if k.startswith("observation.images")))
        print("task:", s.get("task"))
        print("state dim:", tuple(s["observation.state"].shape))
        return 0

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = torch.device(args.device)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = str(device)
    policy = PI05Policy.from_pretrained(args.checkpoint, config=cfg)
    policy.to(device).eval()

    # The base checkpoint ships an EMPTY normalizer (features {}), so the
    # dataset's stats must be injected or the state digits are raw radians
    # pushed through a [-1,1] digitizer -- the F-83 failure exactly.
    # Camera parity, POSITIONAL (docs/runbooks/TRAINING.md). `pi05_base` declares the
    # OpenPI key names and `factory.py:305` only fills input_features when
    # they are empty, so loading the base keeps base_0_rgb/left_wrist_0_rgb/
    # right_wrist_0_rgb and our head/wrist_left/wrist_right never match.
    # This is the same rename the training configs carry.
    rename_map = {
        "observation.images.head": "observation.images.base_0_rgb",
        "observation.images.wrist_left": "observation.images.left_wrist_0_rgb",
        "observation.images.wrist_right": "observation.images.right_wrist_0_rgb",
    }
    expected = {k for k in cfg.input_features if k.startswith("observation.images.")}
    rename_map = {k: v for k, v in rename_map.items() if v in expected}
    if rename_map:
        print(f"renaming cameras -> {sorted(rename_map.values())}")

    pre, _ = make_pre_post_processors(
        cfg,
        args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": rename_map},
            # Mirrors lerobot_train.py:356-378 exactly. `pi05_base` ships an
            # EMPTY normalizer (features {}), so without this the state digits
            # in the prompt are raw radians through a [-1,1] digitizer -- the
            # F-83 failure, and it would look like a finding.
            "normalizer_processor": {
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
                "stats": ds.meta.stats,
            },
        },
    )

    model = policy.model
    pg = model.paligemma_with_expert
    lm = pg.paligemma.model.language_model
    lm_head = pg.paligemma.lm_head
    tok = _tokenizer()

    # Templates. The first is lerobot's REAL prompt; the third is the format
    # reported working in openpi issue #701 (no state digits at all).
    templates = args.template or [
        "Task: {task}, State: {state};\nAction: ",
        "Task: {task}, State: {state};\nSubtask: ",
        "{task}. Subtask: ",
    ]
    results: list[dict] = []

    for i in idxs:
        sample = {k: (v.unsqueeze(0).to(device) if hasattr(v, "unsqueeze") else v) for k, v in ds[i].items()}
        batch = pre(sample)
        task, state_str = _split_prompt(_prompt_of(batch))
        images, img_masks = policy._preprocess_images(batch)

        row: dict = {"frame": i - ep_from, "index": i, "task": task, "state": state_str}
        for tpl in templates:
            text = tpl.format(task=task, state=state_str)
            row[tpl] = _decode(
                model, pg, lm, lm_head, tok, images, img_masks, text, device, args, make_att_2d_masks,
                prepare_attention_masks_4d,
            )
            # The decisive variant: same prompt (state included), but the FAST
            # action band masked out, forcing a TEXT continuation.
            row["BAN " + tpl] = _decode(
                model, pg, lm, lm_head, tok, images, img_masks, text, device, args, make_att_2d_masks,
                prepare_attention_masks_4d, ban=True,
            )
        if args.baseline:
            blank = [torch.full_like(im, -1.0) for im in images]
            row["_blank_images"] = _decode(
                model, pg, lm, lm_head, tok, blank, img_masks,
                templates[-1].format(task=task, state=state_str), device, args,
                make_att_2d_masks, prepare_attention_masks_4d,
            )
        results.append(row)
        banned = row["BAN " + templates[0]]["text"].replace("\n", " ")
        free = row[templates[-1]]["text"].replace("\n", " ")
        print(f"  frame {row['frame']:5d}  real+BAN={banned[:46]!r:50s} nostate={free[:32]!r}")

    _report(results, templates, args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
        print(f"wrote {args.out}")
    return 0


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")


def _prompt_of(batch) -> str:
    """The full pi0.5 prompt string the processor built for this frame."""
    task = batch.get("task") if hasattr(batch, "get") else None
    if isinstance(task, (list, tuple)):
        task = task[0]
    if not isinstance(task, str):
        raise RuntimeError(f"could not recover the prompt string from the batch (got {type(task)})")
    return task


def _split_prompt(prompt: str) -> tuple[str, str]:
    """Recover (task, state_digits) from the processor's assembled prompt.

    The processor builds exactly
    ``f"Task: {task}, State: {state};\\nAction: "`` (processor_pi05.py:74),
    so this inverts that rather than re-deriving the digits independently —
    re-deriving would mean re-implementing the normalizer and the 256-bin
    digitizer, which is how train/eval skew gets introduced (F-63).
    """
    import re

    m = re.match(r"^Task: (.*), State: ([-\d ]*);\n", prompt, flags=re.S)
    if not m:
        raise RuntimeError(f"unrecognised pi0.5 prompt layout: {prompt[:120]!r}")
    return m.group(1), m.group(2)


# openpi maps FAST action tokens as `vocab_size - 1 - fast_skip_tokens - t`
# with fast_skip_tokens=128 over the 256000-entry sentencepiece vocab, i.e.
# 255871 - t. A 2048-entry FAST vocab therefore occupies [253824, 255871].
# Banning that band is what forces a TEXT continuation out of a prompt that
# would otherwise (correctly) answer with actions.
ACTION_BAND = (253824, 255871)


def _decode(
    model, pg, lm, lm_head, tok, images, img_masks, text, device, args, make_att_2d_masks,
    prepare_attention_masks_4d, ban: bool = False,
) -> dict:
    import torch

    def _pick(logits):
        if ban:
            logits = logits.clone()
            logits[:, ACTION_BAND[0] : ACTION_BAND[1] + 1] = float("-inf")
            logits[:, 256000:] = float("-inf")  # <loc*>/<seg*> are not text either
        return logits

    with torch.no_grad():
        enc = tok(text, return_tensors="pt", padding=False, truncation=True, max_length=512)
        tokens = enc["input_ids"].to(device)
        masks = enc["attention_mask"].to(device).bool()

        prefix_embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, tokens, masks)
        att2d = make_att_2d_masks(pad_masks, att_masks)
        dtype = prefix_embs.dtype
        att4d = prepare_attention_masks_4d(att2d, dtype=dtype)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        out = lm.forward(
            inputs_embeds=prefix_embs,
            attention_mask=att4d,
            position_ids=position_ids,
            use_cache=True,
            adarms_cond=None,
        )
        hidden = out.last_hidden_state
        cache = out.past_key_values

        # SELF-CHECK, and it is the load-bearing control for this whole probe.
        # Teacher-forced next-token accuracy over the prompt's own tokens: for
        # each position i, argmax(lm_head(h_i)) should be token i+1. If this is
        # near zero the head is dead, untied, or misaligned and NOTHING below
        # means anything. If it is high, the head works and a boring decode is
        # a real result rather than a bug. openpi issue #701 reports pi0.5
        # dropping the first word or two of a subtask, which is exactly the
        # off-by-one this measurement would catch.
        all_logits = lm_head(hidden.to(lm_head.weight.dtype))
        pred = all_logits[0, :-1].argmax(dim=-1)
        gold = tokens[0, 1:]
        n_lang = gold.shape[0]
        selfcheck = float((pred[-n_lang:] == gold).float().mean().item()) if n_lang else float("nan")

        logits = _pick(lm_head(hidden[:, -1].to(lm_head.weight.dtype)))
        probs = torch.softmax(logits.float(), dim=-1)
        top = torch.topk(probs[0], k=args.topk)
        topk = [
            {"token": tok.decode([int(t)]), "id": int(t), "p": round(float(p), 5)}
            for p, t in zip(top.values, top.indices, strict=True)
        ]

        kv_len = prefix_embs.shape[1]
        pos = int(position_ids[0, -1].item())
        generated: list[int] = []
        nxt = int(torch.argmax(logits[0]).item())
        for _ in range(args.max_new_tokens):
            if nxt == tok.eos_token_id:
                break
            generated.append(nxt)
            emb = pg.embed_language_tokens(torch.tensor([[nxt]], device=device))
            pos += 1
            kv_len += 1
            step_mask = torch.zeros(1, 1, 1, kv_len, dtype=dtype, device=device)
            step = lm.forward(
                inputs_embeds=emb.to(dtype),
                attention_mask=step_mask,
                position_ids=torch.tensor([[pos]], device=device),
                past_key_values=cache,
                use_cache=True,
                adarms_cond=None,
            )
            cache = step.past_key_values
            logits = _pick(lm_head(step.last_hidden_state[:, -1].to(lm_head.weight.dtype)))
            nxt = int(torch.argmax(logits[0]).item())

        in_band = sum(1 for i in generated if ACTION_BAND[0] <= i <= ACTION_BAND[1])
        return {
            "banned": ban,
            "frac_action_tokens": (in_band / len(generated)) if generated else 0.0,
            "text": tok.decode(generated),
            "ids": generated,
            "topk": topk,
            "selfcheck": selfcheck,
            "prompt": text,
        }


def _report(results: list[dict], suffixes: list[str], args) -> None:
    print("\n=== self-check: teacher-forced next-token accuracy, PER TEMPLATE ===")
    print("  A prompt carrying the `State: <digits>` block scores near zero because")
    print("  those ~130 digit tokens are inherently unpredictable as text -- that is")
    print("  a property of the PROMPT, not evidence of a broken head. Judge the head")
    print("  on the highest-scoring template, which is the one without the digits.")
    best = 0.0
    for suffix in suffixes:
        sc = [r[suffix]["selfcheck"] for r in results]
        mean = sum(sc) / len(sc)
        best = max(best, mean)
        print(f"  {mean:.3f}  (min {min(sc):.3f} max {max(sc):.3f})  {suffix!r}")
    if best < 0.15:
        print("  ** No template clears 0.15 -- the lm_head is not predicting text at all.")
        print("  ** Treat every decode below as UNINTERPRETABLE, not as evidence.")
    else:
        print(f"  best template scores {best:.3f} -- the head is aligned and functional.")

    print("\n=== variation across the episode ===")
    keys = [k for k in results[0] if k in suffixes or k.startswith("BAN ")]
    for suffix in keys:
        texts = [r[suffix]["text"] for r in results]
        fa = sum(r[suffix]["frac_action_tokens"] for r in results) / len(results)
        print(f"\n[action-token fraction {fa:.3f}]", end="")
        uniq = sorted(set(texts))
        print(f"\n{suffix!r}: {len(uniq)} distinct continuation(s) over {len(texts)} frames")
        for u in uniq[:10]:
            n = texts.count(u)
            print(f"  x{n:<4d} {u!r}")
        if len(uniq) > 10:
            print(f"  ... and {len(uniq) - 10} more")
        if len(uniq) == 1:
            print("  VERDICT: constant -- the language head carries no per-frame information here.")
        else:
            print("  VERDICT: varies -- the prefix is at least state/image sensitive.")
    if args.baseline and results:
        blank = [r["_blank_images"]["text"] for r in results]
        same = sum(1 for r, b in zip(results, blank, strict=True) if b == r[suffixes[-1]]["text"])
        uniq = sorted(set(blank))
        print(f"\n=== blank-image control (same prompt, cameras set to -1) ===")
        print(f"  {same}/{len(results)} frames decode identically to the real-camera run")
        print(f"  {len(uniq)} distinct string(s) with the cameras blanked:")
        for u in uniq[:5]:
            print(f"    x{blank.count(u):<4d} {u!r}")
        if len(uniq) == 1:
            print("  VERDICT: blanking the cameras COLLAPSES the output to a constant, so the")
            print("  per-frame variation above is driven by the IMAGES, not by the prompt.")


if __name__ == "__main__":
    raise SystemExit(main())
