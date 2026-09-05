"""Track-A gripper dataset edits, written once, applied from both sides.

    from camelo.train.gripper_transforms import GripperTransform

    t = GripperTransform(lead_frames=8, polarity_flip=True)

    actions = t.apply_actions(actions)   # converter, whole episode (N, 20)
    states = t.apply_states(states)      # converter, whole episode (N, 37)

    state = t.encode_state(state)        # eval adapter, per observation
    chunk = t.decode_actions(chunk)      # eval adapter, per emitted chunk

Why this is one module (P2 of GRASP_EXPERIMENT_PROTOCOL.md §1). Every
Track-A edit changes what the dataset says, and every one of them has to
be honoured — or *deliberately* not honoured — at eval. F-63 is what it
costs when that agreement lives in two places: the adapter coerced
37 -> 16-dim proprio at eval while training fed the raw 37, the widths
agreed, nothing complained, and a whole training run was spent on a
mismatch. TRAINING.md states the rule for the gripper specifically —
"encoding the gripper flip twice is how silent train/eval skew happens"
— so it is typed here and imported by the converter and the eval
adapter, never re-typed in either.

**The distinction this module exists to keep sharp.** The Track-A edits
are not one kind of thing. They split in two, and conflating them *is*
the skew:

- **ENCODING** (``Kind.ENCODING``) changes how a value is *represented*.
  A4's polarity flip is one: ``x -> 1 - x`` on the gripper dims, so a
  pi-family fine-tune can reuse its pretrained grasp-direction prior
  instead of overwriting it. Applied at train time, and **exactly
  inverted** at eval — or every action the policy emits is upside down.
- **RELABELING** (``Kind.RELABELING``) changes *what the policy is
  taught to output at time t*. A1's lead is one: the gripper action at
  frame t is replaced by the gripper action at frame t+k, so the close
  is supervised earlier and the action stops being a copy of the state.
  Applied at train time **only**. It has no eval-side inverse, and
  undoing it at eval would exactly cancel the experiment — the policy
  would emit its close k frames early and the adapter would politely put
  it back where it started.

That split is structural here, not a comment. Every knob is declared in
``KNOBS`` with its ``Kind``; ``is_identity``, ``to_dict`` and
``provenance`` are derived from that table; and ``decode_actions``
reaches its work through ``encoding_only()``, whose relabeling knobs are
reset to off — so a relabel can never grow an eval-side inverse by
accident. A field added to the dataclass without an entry in ``KNOBS``
raises at construction rather than quietly picking a side.

The canonical gripper channel is an open fraction, frozen with its
measurement in docs/CONTRACTS.md (F-18) — that is the only place it is
defined and nothing here changes it. A4 is a transform layered on top,
inverted before anything reaches the wire, exactly the two-directional
pattern ``camelo/policy/adapters/droid8.py`` already runs for DROID
checkpoints.

**Order of operations, pinned once:** ``apply_actions`` **relabels, then
encodes** (A1 then A4). Decide *what* to teach in canonical units, then
decide *how* to write it down. The two happen to commute — a gather
along time, and an elementwise ``1 - x`` on the same column — so the
wrong order is not a silent skew today; the order is pinned anyway so
the next knob has a defined slot.

**Where it goes in the pipeline.** Apply on the canonical 37-dim state
and 20-dim action, *before* ``camelo.train.convert_model_state``: the
model16 gripper dims (7 and 15) then inherit the flip, and applying it
again there would be the F-63 double-encode in a new costume. On the
eval side the adapter must call ``encode_state`` on the 37-dim
observation before it packs, and ``decode_actions`` on the chunk before
anything executes. Recompute + repair + verify stats after any dataset
edit made with this (``camelo/train/dataset_stats.py``; F-83 is what
skipping it costs).

**Scope: A1 and A4 only.** Both are per-frame value edits over a fixed
frame set, which is precisely what a converter and an adapter can both
perform on the same footing.

- **A2 is retired** — measured, this corpus's gripper action is already
  exactly {0.0, 1.0} with no frames in transition, so a binarizer would
  change nothing. There is deliberately none here; un-retire A2 first if
  a corpus ever records a genuinely continuous gripper.
- **A3 (upweight the transition frames) is a SAMPLER** — it changes
  which windows are drawn, not the values inside them. It belongs in the
  training sampler / dataset index.
- **A5 (idle filter + rate subsample) is FRAME SELECTION** — it changes
  the frame set itself, and with it timestamps, episode lengths and
  stats. It belongs in the converter's frame-selection pass.

  Neither has a per-frame eval-side counterpart, so neither is an
  encoding or a relabeling in the sense above; putting either here would
  blur the one distinction this module is for.

Numpy + stdlib + ``camelo.contracts`` only, on purpose: ``camelo/train/``
must stay importable with no torch, no lerobot and no ROS, and
``make test`` runs with none of them installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path

import numpy as np

from camelo import contracts as C

# The only columns any transform here may touch. Everything else in the
# vector must come out byte-identical — that is a tested property.
STATE_GRIP_DIMS: tuple[int, ...] = (C.S_LEFT_GRIP, C.S_RIGHT_GRIP)
ACTION_GRIP_DIMS: tuple[int, ...] = (C.A_LEFT_GRIP, C.A_RIGHT_GRIP)


class Kind(Enum):
    """Which side of the train/eval boundary a knob is allowed to live on.

    ENCODING   applied at train AND exactly inverted at eval.
    RELABELING applied at train ONLY; no eval-side inverse exists, and
               inventing one cancels the experiment it implements.
    """

    ENCODING = "encoding"
    RELABELING = "relabeling"


@dataclass(frozen=True)
class _Knob:
    """One flag on `GripperTransform`, and the side of the fence it is on."""

    field: str
    kind: Kind
    experiment: str
    off: object  # the value that means "this knob does nothing"
    doc: str


#: Every knob, with its kind. This table — not the method bodies — is what
#: decides whether eval inverts a transform or ignores it.
KNOBS: tuple[_Knob, ...] = (
    _Knob(
        field="lead_frames",
        kind=Kind.RELABELING,
        experiment="A1",
        off=0,
        doc="gripper action[t] <- gripper action[t + k]: the close moves k frames earlier",
    ),
    _Knob(
        field="polarity_flip",
        kind=Kind.ENCODING,
        experiment="A4",
        off=False,
        doc="x -> 1 - x on the gripper dims of state and action (pi-family prior reuse)",
    ),
)


def _frames(array, width: int, what: str) -> np.ndarray:
    """(N, width) float32 COPY, or a message naming the width we wanted."""
    out = np.array(array, dtype=np.float32, copy=True)
    if out.ndim != 2 or out.shape[1] != width:
        raise ValueError(f"expected (N, {width}) {what}, got {np.shape(array)}")
    return out


@dataclass(frozen=True)
class GripperTransform:
    """The Track-A gripper edits, as one value you can log and re-read.

    Construct it once from config, hand the same object to the converter
    and to the eval adapter, and write `provenance()` next to both the
    dataset and the checkpoint.
    """

    lead_frames: int = 0  # A1, RELABELING: action[t] <- action[t + lead_frames]
    polarity_flip: bool = False  # A4, ENCODING: x -> 1 - x on the gripper dims

    def __post_init__(self) -> None:
        unclassified = {f.name for f in fields(self)} - {k.field for k in KNOBS}
        if unclassified:
            raise TypeError(
                f"gripper knob(s) {sorted(unclassified)} declared without a Kind — add them "
                "to KNOBS as ENCODING (inverted at eval) or RELABELING (train only). A knob "
                "with no declared side of the train/eval boundary is exactly the F-63 skew "
                "this module exists to prevent."
            )
        lead = self.lead_frames
        if isinstance(lead, bool) or not isinstance(lead, (int, np.integer)):
            raise ValueError(f"lead_frames must be a whole number of frames, got {lead!r}")
        if lead < 0:
            raise ValueError(
                f"lead_frames must be >= 0, got {lead} — a negative lead would make the "
                "action LAG the state, which is the copycat failure A1 exists to break"
            )

    # ---- construction / serialization ------------------------------------

    @classmethod
    def identity(cls) -> GripperTransform:
        """The no-op transform: what an untransformed dataset was built with."""
        return cls()

    @classmethod
    def from_dict(cls, payload: dict | None) -> GripperTransform:
        """Rebuild from a provenance/config dict; `None` means identity.

        Unknown keys are refused rather than ignored: a misspelled knob
        that silently trains the identity is a whole wasted run.
        """
        if payload is None:
            return cls.identity()
        if not isinstance(payload, dict):
            raise ValueError(
                f"gripper transform payload must be a dict or None, got {type(payload).__name__}"
            )
        known = {k.field for k in KNOBS}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown gripper transform key(s) {unknown} — known keys are {sorted(known)}"
            )
        flip = payload.get("polarity_flip", False)
        if not isinstance(flip, bool):
            raise ValueError(f"polarity_flip must be a bool, got {flip!r}")
        return cls(lead_frames=payload.get("lead_frames", 0), polarity_flip=flip)

    def to_dict(self) -> dict:
        """Plain JSON-safe values, one per knob, including the inactive ones."""
        return {k.field: type(k.off)(getattr(self, k.field)) for k in KNOBS}

    @property
    def is_identity(self) -> bool:
        """True when every knob is off, so applying this changes nothing."""
        return all(getattr(self, k.field) == k.off for k in KNOBS)

    def encoding_only(self) -> GripperTransform:
        """This transform with every RELABELING knob reset to off.

        The eval side works through this, so a relabel cannot acquire an
        inverse by someone reading the wrong flag in a decode path.
        """
        return replace(self, **{k.field: k.off for k in KNOBS if k.kind is Kind.RELABELING})

    # ---- the two halves, in place on a copy ------------------------------

    def _encode_grip(self, out: np.ndarray, dims: tuple[int, ...]) -> None:
        """ENCODING knobs. Shared by both directions on purpose: `1 - x` is
        its own inverse, so encode and decode are one body and cannot drift
        apart (the `droid8.py` two-sided flip, factored). A future encoding
        knob that is NOT an involution needs its own inverse in
        `_decode_grip` — that seam is why the two methods exist."""
        if not self.polarity_flip:
            return
        for dim in dims:
            out[..., dim] = 1.0 - out[..., dim]

    def _decode_grip(self, out: np.ndarray, dims: tuple[int, ...]) -> None:
        """Exact inverse of `_encode_grip`."""
        self._encode_grip(out, dims)

    def _relabel_actions(self, out: np.ndarray) -> None:
        """RELABELING knobs. Train side ONLY — nothing on the eval path may
        call this, and nothing on the eval path may undo it.

        A1: `action[t] <- action[t + k]` on both gripper dims, so a close at
        frame f is taught at frame f-k. The last k frames hold the episode's
        final value — no wrap (which would teach a close at the end from the
        start) and no dropped frames (which would desynchronise the episode
        from its states, images and timestamps).
        """
        lead = int(self.lead_frames)
        if lead == 0:
            return
        n = out.shape[0]
        source = np.minimum(np.arange(n) + lead, n - 1)
        for dim in ACTION_GRIP_DIMS:
            # Advanced indexing copies before assigning, so this is safe
            # in place even though source and destination overlap.
            out[:, dim] = out[source, dim]

    # ---- dataset / train side (whole-episode arrays) ---------------------

    def apply_states(self, states: np.ndarray) -> np.ndarray:
        """(N, 37) canonical recorder state -> transformed copy.

        Encoding-only, and that is a design statement rather than an
        omission: A1 moves the ACTION relative to the state, which is the
        whole point of it. `encode_state` reproduces exactly this.
        """
        out = _frames(states, C.STATE_DIM, "canonical states")
        self._encode_grip(out, STATE_GRIP_DIMS)
        return out

    def apply_actions(self, actions: np.ndarray) -> np.ndarray:
        """(N, 20) canonical action -> transformed copy. Applies BOTH classes.

        Order: RELABELING first (decide what to teach, in canonical units),
        then ENCODING (decide how to write it down).
        """
        out = _frames(actions, C.ACTION_DIM, "canonical actions")
        self._relabel_actions(out)  # A1 — train only, never inverted
        self._encode_grip(out, ACTION_GRIP_DIMS)  # A4 — inverted at eval
        return out

    # ---- eval side -------------------------------------------------------

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """(37,) or (N, 37) -> copy with the ENCODING transforms applied.

        Matches exactly what `apply_states` wrote into the dataset, because
        it is the same call. Feed the result to the adapter's packer.
        """
        out = np.array(state, dtype=np.float32, copy=True)
        if out.ndim not in (1, 2) or out.shape[-1] != C.STATE_DIM:
            raise ValueError(
                f"expected ({C.STATE_DIM},) or (N, {C.STATE_DIM}) canonical state, "
                f"got {np.shape(state)}"
            )
        self.encoding_only()._encode_grip(out, STATE_GRIP_DIMS)
        return out

    def decode_actions(self, actions: np.ndarray) -> np.ndarray:
        """(N, 20) model output -> canonical copy.

        The exact inverse of the ENCODING half, and nothing else. The
        RELABELING half has no inverse here by construction: undoing A1's
        lead would shift the policy's close back to where the untransformed
        data would have put it and cancel the experiment outright. A
        lead-only transform therefore decodes as the identity, on purpose.
        """
        out = _frames(actions, C.ACTION_DIM, "model actions")
        # Through encoding_only(), so the relabeling knobs are not merely
        # unread here — they are off in the object doing the reading.
        self.encoding_only()._decode_grip(out, ACTION_GRIP_DIMS)
        return out

    # ---- what was done, for the record -----------------------------------

    def provenance(self) -> dict:
        """For the dataset provenance JSON and the checkpoint sidecar.

        Split by kind, so whoever wires the eval side a month from now can
        read off which half they owe an inverse and which half they must
        leave alone.
        """
        config = self.to_dict()
        active = {kind: {} for kind in Kind}
        for knob in KNOBS:
            if config[knob.field] != knob.off:
                active[knob.kind][knob.field] = config[knob.field]
        return {
            "module": __name__,
            "config": config,
            "identity": self.is_identity,
            "encoding": active[Kind.ENCODING],
            "relabeling": active[Kind.RELABELING],
            "experiments": sorted(
                {k.experiment for k in KNOBS if config[k.field] != k.off}
            ),
            "gripper_dims": {
                "state": list(STATE_GRIP_DIMS),
                "action": list(ACTION_GRIP_DIMS),
            },
            "note": (
                "encoding: applied at train AND inverted at eval "
                "(encode_state / decode_actions). relabeling: applied at train ONLY "
                "and never inverted — undoing it at eval cancels the experiment."
            ),
        }


# --- the sidecar -----------------------------------------------------------
#
# Nothing in a checkpoint records which gripper transform trained it. The
# weights of a polarity-flipped run are byte-shaped exactly like an unflipped
# one, so a flipped checkpoint loaded without its transform emits every
# gripper command upside down -- no error, no shape mismatch, just a policy
# that opens to grasp. That is F-63's shape again, and the fix is the one
# `pi05_state_route.py` already proved: the fact travels beside the run and
# is re-applied at load.
#
# An ABSENT sidecar means the identity, so every checkpoint trained before
# this module existed evaluates exactly as it did before. The known cost of
# that choice (`scripts/eval_recipe.py`) is that "absent" and "explicitly
# identity" read the same, and a checkpoint copied without its sidecar
# silently becomes the former -- so `read_sidecar` is paired with
# `sidecar_is_explicit` for callers that need to tell the two apart.
SIDECAR = "camelo_gripper_transform.json"


def sidecar_path(run_dir: Path) -> Path:
    """A SIBLING file, deliberately not ``run_dir/SIDECAR``.

    lerobot refuses to start when its output dir already exists ("already
    exists and resume is False"), so creating the run dir early to hold the
    sidecar makes every run fail before step 0. Writing beside the dir keeps
    the marker available from the moment the job launches -- including for a
    run that dies mid-training -- without touching the path lerobot owns.
    Same reasoning, same layout, as ``pi05_state_route.sidecar_path``.
    """
    return run_dir.parent / f"{run_dir.name}.{SIDECAR}"


def write_sidecar(run_dir: Path, transform: GripperTransform) -> None:
    """Record the transform beside a training run, for the eval side to find."""
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path(run_dir).write_text(
        json.dumps(transform.provenance(), indent=2) + "\n"
    )


def _find_sidecar(checkpoint: Path) -> Path | None:
    for parent in [checkpoint, *checkpoint.parents]:
        for candidate in (parent / SIDECAR, sidecar_path(parent)):
            if candidate.is_file():
                return candidate
    return None


def sidecar_is_explicit(checkpoint: str | Path) -> bool:
    """True when a sidecar was actually found, rather than defaulted away.

    ``read_sidecar`` cannot distinguish "trained with the identity" from
    "sidecar lost in an rsync"; this can, so a caller that would be wrong
    either way can say so out loud instead of guessing.
    """
    return _find_sidecar(Path(checkpoint)) is not None


def read_sidecar(checkpoint: str | Path) -> GripperTransform:
    """Walk up from a checkpoint dir to the run root looking for the sidecar.

    Absent sidecar means the identity -- see the note above.
    """
    found = _find_sidecar(Path(checkpoint))
    if found is None:
        return GripperTransform.identity()
    payload = json.loads(found.read_text())
    # Accept both a bare config and a full provenance() blob, because the
    # sidecar on disk is the latter and a hand-written one will be the former.
    return GripperTransform.from_dict(payload.get("config", payload))
