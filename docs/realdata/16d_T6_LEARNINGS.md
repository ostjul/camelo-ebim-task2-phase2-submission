# T6 learnings and next steps — day-3 wrap-up, state at 2026-09-03 end of day

Companion to [16](16_RIG_TEST_PROTOCOL.md) (evidence rows R-62…R-86, open
questions U-21…U-42) and [16c](16c_T6_CHEATSHEET.md) (the operator runbook).
Everything below is re-derivable from those rows and from
`outputs/rig/munich_2026-09-01/runs.csv` / `rollouts.csv` / the archived run
directories.

## 1. What is proven

**The closed-loop remote-policy pipeline works on the robot.** Five 30 s
rollouts of the served ACT checkpoint on the right arm (R-67, R-70 ×3, R-71),
all with the same signature:

| quantity | value | where |
|---|---|---|
| activation publish cadence | 20 Hz (gate ≥ 8 Hz) | U-28 |
| warm-up before the switch | reset 0.13 s, first inference 0.6 s, chunk discarded | U-28 |
| activation → first command | 0.56 s (keep-alive covers it, max interval 0.10 s) | U-28 |
| control ticks in 30 s | 587 (20 Hz; the sync loop managed 90) | U-29 |
| chunks / inferences | 70, median 0.28 s (server 0.03 s, transport 0.26 s) | U-29 |
| starved / dropped / stale chunks | 0 / 0 / 0 (except R-71, see §3) | U-29 |
| joint-state age | median 0.008 s, max 0.024 s | U-27 |
| image ages | 0.06–0.14 s, all three cameras 600 frames at 19.99 Hz | U-23 |
| deactivation | self, `switch_controller deactivate ok`, confirmed | — |
| companion afterwards | 4 controllers, 0 FATAL, 0 `Rejecting GELLO`, 2 gripper clients | — |

The parity gate is met on the serving box (A = 0.0 on all 15 dims, the
`camera_blank` control has teeth), the gripper unit chain is correct end to end
(0..1 open fraction on the wire), and the base/table placement is reproducible
by head-camera overlay (mean pixel difference 30.6 vs episode 163's frame 0;
≈95 when a metre off).

**Also proven by end of day 3:**

- **The wire is proven with recorded frames, not just assumed correct by code
  review** (R-77): episode 163's own recorded frames sent through the rig's
  own client path agree with the demo's own actions to 0.03–0.05 rad — an
  order of magnitude tighter than the live-observation gap — so the
  client→server wire and arm path are cleared as a cause of anything seen
  since.
- **All three site confounds are removed, measured, not assumed:** wrist
  resolution 848×480 → 640×480 (U-21, R-73), companion `max_goal_velocity`
  0.5 → 1.0 rad/s (U-26, R-74), and wrist auto-exposure → fixed exposure 5000
  matching the recording (U-37, R-79c). None of the three moved the outcome
  once isolated — U-26 never bound, U-21 and U-37 corrected real drift but the
  policy still failed to close afterward (R-74 onward, and R-80).
- **Every executor option this window could try has now been measured:**
  wall-clock `index` splice (R-67), `nearest` splice (R-71, wrong direction),
  leash-to-measured (R-72/R-72b/R-74), ensembling (U-33, PARKED — not enough
  chunk overlap at the 21-row horizon and 0.45 s RTT to matter), and
  synchronous plan-then-execute (R-78, R-80). None of them, alone, produced a
  repeatable grasp — see §3.

## 2. What went wrong, in order, and what fixed it

| when | symptom | mechanism | fix | status on rig |
|---|---|---|---|---|
| 09-02 17:29 (R-64) | right controller FATAL 0.5 s after activation, whole stack down | command stream not fresh in the post-activation window: 4 publishes/0.6 s at activation, reset + 1.44 s first inference after the switch, keep-alive ~5 Hz on the shared executor; joint-state stream then frozen for 30 s unnoticed | U-27 joint-state liveness guard; U-28 keep-alive thread + cadence gate + warm-up before the switch + publisher depth 1 | measured OK ×5 |
| 09-02 17:29 | 90 ticks in 30 s | synchronous inference blocked the 20 Hz loop for the 0.45 s round trip | U-29 async inference (hold last target on starvation) | measured OK ×5 |
| 09-02 18:14 (R-66) | websocket handshake timeout before any participant | ssh tunnel transport died idle (local listener kept accepting); server had been replaced by the colleague | U-30: `ServerAliveInterval` on the tunnel, dummy client before every rollout | procedure |
| 09-03 09:38 (R-67/R-68) | arm reaches the approach region, then hovers with a few-Hz wiggle, never closes | every async chunk installed at the wall-clock row (≈ 9–11 of 21) → a maximal 0.05 rad clamped jump on 70/70 arrivals | U-31 `--chunk-splice nearest` | measured WRONG WAY (R-71) |
| 09-03 10:17 (R-71) | executor held 70 % of ticks, all splice shifts positive | the chunk is anchored at the measured pose, the splice searched for the commanded target (which leads it); near-stationary chunks make `argmin` arbitrary | U-32 leash command to `measured ± 0.15 rad`, splice anchored to measured, forward search bounded, tie-break toward the wall row | CLOSED, measured 10:55 (R-72): hold gone (starved 16/579), splice shifts back to R-71's magnitude; new disturbance — a 0.15 rad sawtooth at chunk cadence, = the leash width; binding constraint now U-26 |
| 09-03 11:15 (U-21) | live wrist cameras stayed 848×480 despite passing `rgb_camera.color_profile` / a bare `depth_module.color_profile` launch argument | stock `rs_launch.py` 4.55.1 silently drops any launch argument the launch file does not declare in `configurable_parameters` — no error, no log line | pass the resolution through `config_file:=configs/d405_color_640x480.yml` instead of a bare argument | CLOSED, verified `Width: 640, Height: 480` on both wrists (R-73); carried into every later rollout with `--allow-camera-shape-mismatch` dropped |
| 09-03 10:55–11:06 (R-72/R-72b, U-32/U-26) | hold gone under the leash, but `arrival_jump_rad` did not drop — a 0.15 rad sawtooth appears at chunk cadence | the leash's own re-anchoring at every chunk hand-over produces a jump equal to the leash width; later shown (R-74's chunk dump, U-33) to be a policy/executor limit cycle upstream of both the leash and the velocity cap | none found by site tuning — explained, not fixed; see §3 for what actually changed the outcome | unresolved by tuning; superseded as a live symptom once R-78/R-80 moved to synchronous execution |
| 09-03 12:01–12:11 (R-77/R-76) | live pad row shifted and reordered vs the corpus (75 px spacing, RED in slot 3) vs episode 163 frame 0 (97 px spacing, RED in slot 1) | the mean-pixel-difference overlay gate only catches gross base/table misplacement, not a shifted/reordered pad row within an otherwise-plausible-looking frame | R-77's wire probe caught it by comparing against the corpus row directly; pad corrected to match episode 163 frame 0 | CLOSED — R-76 re-ran the identical line on the corrected scene and the chunk #1 row-0-vs-demo offset was unchanged, ruling the pad layout out as a cause of the hover |
| 09-03 12:23–12:30 (R-79/R-79c, U-37) | live wrist images badly overexposed vs the recording — mean RGB ≈ 222/198/196 vs recorded 121/124/120 (left); the table rendered as a near-white sheet | D405 wrist auto-exposure on the live rig disagreeing with the recording session's exposure | `depth_module.enable_auto_exposure: false` + `depth_module.exposure: 5000` in `configs/d405_color_640x480.yml` on both wrist launches | CLOSED — live wrist mean RGB now matches the recording (122/123/120 left, 99/100/98 right, 0 % saturated); necessary but **not sufficient**: R-80, run with this fix in place, still produced 3/3 ABSENT rollouts |

Other facts that cost time: the companion clock drifts ~10 ms/min (sync with
the arms down, every ~30 min); a `--restart` can leave an orphan
`ros2_control_node` (the four numbers read 5 0 2 1; kill the old-etime one);
`docker ps`'s default command column truncates `serve_policy` away (use
`{{.Names}}`); the EC2 public IP changes on every instance restart; the spine
was at a constant 434 mm in every recorded frame (R-69; live value not yet
read — `read_spine.py`); `ros2 launch` silently drops any launch argument the
launch file does not declare in `configurable_parameters` — no error, no log
line — which is why both `rgb_camera.color_profile` and a bare
`depth_module.color_profile` argument were no-ops on the D405 wrists and the
working fix had to go through `config_file` instead (U-21, R-73).

## 3. What the policy did (all five runs, pre-fix executor)

Right arm from episode 163's frame 0 toward the corpus' pre-grasp region within
≈10 s, then hover for the remaining 20 s; the hover pose varies run to run;
gripper command never below 0.86 (four of five runs ≥ 0.99). `clamped_pct`
27–32 % and `max_requested_delta` 0.22–0.41 rad/tick: the policy asks for
5–8× the clamp. Verdict by mechanism (15 §2.4): ABSENT. **Not yet a statement
about the checkpoint** — three confounds remain:

1. the executor's own chunk hand-over disturbance (U-31/U-32, being fixed);
2. wrist images off-contract, 848×480 instead of 640×480 — **CLOSED 2026-09-03
   11:35 (U-21, R-73)** via `config_file:=configs/d405_color_640x480.yml` on
   both wrist launches, not `enable_depth:=true`; all five runs above were
   still run under `--allow-camera-shape-mismatch` and predate the fix;
3. ~~the companion's `max_goal_velocity` 0.5 rad/s~~ — **REMOVED 2026-09-03
   ~11:25 (U-26, R-74)**: the site raised the cap to 1.0 rad/s in both
   `controllers.yaml` copies; the re-measured hover speed p95 is 0.5–0.6 rad/s
   (max 0.8), nowhere near the new ceiling, and the sawtooth is unchanged —
   the velocity cap was never the mechanism.

With both site confounds closed, R-74 (11:28:59, `t6_act_112859`, branch
`0748bcd`, run under `--chunk-dump`) is the first rollout that says something
about the executor/checkpoint on its own terms; R-75 (`--chunk-splice
index`, everything else as R-74) repeated it and confirmed the executor
choice does not matter — same hover, same L2 distance to the demo (0.68 vs
R-74's 0.82). **Current best explanation of the hover (U-35, OPEN, narrowed
2026-09-03 12:xx by R-77/R-76):** in both R-74/R-75 dumps, chunk #1 —
computed at the verified start pose (0.0003 rad from episode 163 frame 0,
the real scene) — disagrees with the demonstration's own frame-0 action by
up to 0.2 rad on several joints, while the chunk's later rows agree with the
demo closely; the earlier U-34 reading ("chunk centred on the state") was
partly an artefact of the `nearest` executor holding the arm at the chunk
centre — under `index` (R-75) the offset grows instead of shrinking.

**The wire and arm path are now cleared, and the scene layout is ruled out
as the cause.** R-77 (12:01) fed episode 163's own recorded frame 0 through
the rig's own client path (`RemoteBackend` → tunnel → served ACT → adapter)
and got row-0 agreement with the demo to 0.03–0.05 rad — an order of
magnitude tighter than R-74/R-75's 0.15–0.2 rad live-observation gap — and a
same-day code audit found no mismatch in the image/wrench pipeline either.
R-77 also caught a real scene defect (the live pad row shifted and
reordered: 75 px spacing, RED in slot 3, vs the corpus's 97 px spacing, RED
first) that the mean-pixel-difference overlay gate had not detected; the
scene was corrected and R-76 (12:11) re-ran the same T6-6 line — **the
chunk #1 row-0-vs-demo offset was unchanged** (`[+0.17,0,−0.18,+0.01,−0.05,
+0.22,+0.17]` vs R-75's `[+0.19,0,−0.18,+0.02,−0.02,+0.19,+0.18]`), so the
pad layout was not the (or not the whole) cause of the hover.

**U-35 narrowed to two remaining suspects, both never compared at
pixel/value level on the rig: (a) live image appearance vs the recordings**
(lighting, shadow, exposure, colour; the wrist views in particular had
never been checked) **and (b) the live wrench values** (never printed on
the rig, only ever assumed correct by code review). The handoff probe shows
the same checkpoint reproduces held-out actions teacher-forced, so with the
proprioceptive state and the wire both verified correct, one of these two
remained the gap between the live scene and what the checkpoint was trained
on.

**R-78 (12:15:58) was the best executor at that point.** Same scene, same
wrist/slew config, same U-35 offset (`[+0.16,0,−0.18,+0.01,−0.04,+0.22,
+0.16]` — still unchanged), but run through the SYNCHRONOUS plan-then-execute
loop instead of the async one: `--no-async-inference --chunk-time-base
arrival --chunk-splice index --replan-steps 10` (observe → infer, arm holds
≈0.35 s under the keep-alive → execute the new chunk's own rows 0–9 verbatim
→ observe again). 369 ticks in 30 s, 33 chunks, `clamped_pct` 21.1 %,
`arrival_jump_rad` mean 0.128/max 0.291. Where every one of R-67…R-76 (six
async rollouts) hovered with the gripper command ≥ 0.63, R-78 produced the
**first full task sequence**: approach (0–7 s) → close (8–19 s) → transport
to near the demo's own post-grasp pose (24 s) → hold closed. Grasp/pad-contact
closed on air, nowhere near the pad (operator report) and U-35's observation
gap did not go away — so this was a mechanism/executor result, not a
checkpoint verdict, and it opened **U-36**: is the sequence caused by the
sync/async split itself, by verbatim (no-skip) row execution, or by the j1
ratchet that R-78 also shows? See 16 R-78/U-36 for the full detail.

**U-35 CLOSED 12:23–12:28 (R-79) — the row-0 action gap decomposes into
three additive, independent causes, none of them a wire, splice, or
scene-layout bug.** A live capture at the verified start pose (all 14 arm
joints within 0.0002 rad of episode 163 frame 0; `right_gripper_open` 0.856,
not the recorded 1.000) plus six sends mixing recorded/live images and state
through the same client path (`run_dummy_client.py --capture-live` /
`--images-from`/`--state-from`) isolate each input's own contribution to the
row-0 `|diff|` on right j1..j7:

| input held live, rest recorded | joints moved | likely cause |
|---|---|---|
| head image | j1 +0.13, j7 +0.12 | background, a shadow band, and the teleoperator visible at the right edge of every corpus head frame (U-38) |
| wrist images | j6 +0.15 (≈⅓ from exposure alone) | live wrists badly overexposed — mean RGB ≈ 222/198/196 vs recorded 121/124/120 (left), 192/198/196 vs 94/97/92 (right) — plus a different composition, table vs floor+base (U-37) |
| wrench + gripper state | j7 +0.16, j3 +0.06 | live wrenches numerically far from the recording's; gripper reading 0.856 vs the recorded 1.000 (U-39) |

No left/right wrist mix-up (swapping the wrist images reproduces the same
numbers to 4 decimals). **The checkpoint has memorised the recording
session's appearance and sensor biases; no executor or splice change can fix
that** — the remaining lever is either bringing the rig closer to the
recording (U-37 wrist exposure, U-38 head background/shadow/operator
position, U-39 wrench bias vs the companion's payload configuration at
recording time) or retraining the policy with augmentation. Full probe data
in 16 R-79; the three follow-ups are 16 U-37/U-38/U-39.

U-37 (wrist exposure) closed same-day (R-79c: fixed exposure 5000 puts the
live wrists on the recording) and U-39 (wrench bias) was answered offline
(payload fit: the corpus bias is a per-episode re-zero, not a payload/gravity
effect; the live at-rest wrench sits within 0.7σ of the corpus under the
checkpoint's own normaliser — in distribution, no correction applied). A
calibration pass (teacher-forced frame-0 spread across held-out corpus
episodes: mean 0.03–0.07 rad, p90–max 0.08–0.19 rad on the same joints U-35
moves) shows the rig's 0.13–0.2 rad gap sits at the tail of the checkpoint's
own noise, not off the scale — episode 163 is a typical frame-0 draw.

**R-80 (13:47–13:56, block B2, the first SCORED block) is the verdict this
window can support: the checkpoint is session-sensitive, and R-78's sequence
was one accidental draw, not a fix.** With U-37's exposure correction in
place (the one observation change since R-78) and the gripper fully
re-opened, three rollouts on the frozen slot sequence (seq 0/1/2 = x048/x061/
x048) all wiggled and never descended — worse than R-78, not better — and the
operator stopped the block after r02 rather than spend the remaining 12 on a
policy that was visibly not going to close. **Across the 12 closed-loop
rollouts run on the robot this window (R-67, R-70 ×3, R-71, R-72, R-72b,
R-74, R-75, R-76, R-78, and R-80's 3), there is zero grasp and exactly one
accidental full task sequence (R-78) that did not reproduce** when the next
opportunity came around. That is the shape of a checkpoint that has
memorised session-specific nuisance variables rather than learned the task:
sometimes those variables line up by chance (R-78) and the arm executes the
motion in joint space with no relation to the pad (operator report,
confirmed by R-79's decomposition); usually they do not, and it hovers or
wiggles. See §4 for what that implies for tomorrow.

**Update, 14:12–14:58 (R-81/R-82/R-83) — this pessimism does not survive the
next three hours: the untested combination (async, observation-time base,
offset splice) reaches the demo's grasp pose, and it was never the
sync/async choice that mattered.** Two threads run in this window: a second
checkpoint (VLA-JEPA @30000, EC2 port 8766) and further ACT executor variants
(ACT @100000, EC2 port 8767).

- **VLA-JEPA (R-81, v00/v01): mechanism PASS ×2, task ABSENT ×2, and not
  comparable to any ACT number above.** The wire probe agrees with the demo
  to ≤ 0.04 rad at the recorded frame — VLA-JEPA is not off-distribution the
  way ACT was (U-35). Its failure is a transport/horizon mismatch instead:
  the checkpoint's own action chunk is 7 rows (0.35 s) at 20 Hz, shorter than
  the measured 0.37 s server round trip. Synchronous execution (v00) jumps
  hard and freezes within 2 s (L2 to demo f200 min 1.83); async with the
  arrival-time base (v01) starves every chunk and re-commands stale poses one
  round trip old — the operator's "moves, moves back to an earlier joint
  state, moves again." New open question **U-40**; a transport fix
  (client-side resize to 224 px, two cameras) is in flight.
- **ACT, async control for U-36 (R-82, `t6a_a00_142947`, 14:29):** async with
  `nearest` splice + observation-time base reaches L2 0.21 to the demo's
  grasp pose at 26 s — an async loop approaches the pad where the
  synchronous line (R-78, R-80) never did. **U-36 ANSWERED: the synchronous
  stop-and-go loop is off-distribution for this checkpoint; U-37's
  wrist-exposure fix was never the regression R-80 seemed to show.** All
  three of §3's candidate explanations for R-78 ((i) stationary-arm
  observation, (ii) verbatim row execution, (iii) the j1 ratchet) are now
  moot — R-78's sequence was an accidental draw under a since-abandoned
  executor line, not evidence that synchrony itself helps.
- **ACT, offset splice (R-83, `t6a_a01_145115` / `t6a_a02_145250` /
  `t6a_a02_145333` / `t6a_a02_145428`, 14:51–14:58):**
  `--async-inference --chunk-time-base observation --chunk-splice offset
  --splice-ramp-ticks 20 --replan-steps 8 --max-delta 0.04`. The offset
  splice carries the hand-over discontinuity as an additive correction bled
  off over 1 s instead of an index jump, and observation-time indexing stops
  re-committing to rows the arm has already passed. `starved_ticks` 0 on all
  four runs, `clamped_pct` ≤ 1 % on three of four (a01 at 9.4 %, still
  settling in). The 120 s repeat (`t6a_a02_145428`) reaches **L2 to the
  demo's f200 grasp pose = 0.08 rad at t = 24 s** — an order of magnitude
  closer than any run in §5's table — with the gripper command closing to
  0.20 (the demo's own closed value, measured 0.02). Operator: "definitely
  the best"; the 120 s run's physical outcome (pad picked up or not) is
  **pending** the operator's visual report. **This is now the best executor
  line measured on this rig, on either checkpoint, by a wide margin.**

**Update, 14:58–15:54 (R-84/R-85/R-86) — the approach is now reliable; the
close is the checkpoint's coin flip.** Three more probes closed out day 3:

- **VLA-JEPA (R-84, v02/v03): the transport fix landed but the link is
  latency-bound, not payload-bound.** Client-side resize to 224×224 + a
  two-camera subset cut the payload 117 kB → 17 kB, but RTT only dropped
  0.24 s → 0.225 s — network latency plus server compute is the floor, and it
  still exceeds VLA-JEPA's 0.35 s chunk horizon. v02 wandered (L2 min 1.18);
  v03 crawled ("basically did not move" — the executor holds most of every
  tick, since 0.22 s of the 0.35 s chunk is already gone at arrival). **U-40
  is superseded by U-41: VLA-JEPA needs a server co-located with the robot,
  not another client-side trim.** Not a checkpoint verdict.
- **ACT, offset splice, third repeat (R-85, `t6a_a03_150950`): reaches the
  grasp pose AND closes on it.** Same a02 line as R-83, `--seconds 120`. L2
  to demo f200 down to **0.09 rad at 48 s**, gripper closes to measured 0.02
  at the grasp pose (right j5/j6 −2.06/2.65 vs demo −2.09/2.60) — then
  re-opens to half-open and hovers there for 90 s. Operator: "really good,
  even the right gripper position; the pad was between the fingers, but the
  full close came after it had moved away again" — the ≈3 s close ramp (the
  demos close in ≈1 s) lets the arm drift off the pad before the close
  finishes. **The approach/pose problem (U-35, U-36) is solved; what remains
  is gripper timing, not pose.**
- **Gripper latch, first rig test (R-86, a04/a05): a threshold-only latch
  cannot tell the pre-shape dip from the real close.** `--gripper-latch`
  (`ec62d7f`) engages once the command drops below a threshold and holds. a04
  engaged on the pre-shape dip (0.399, 9.7 s) while still 0.22 rad from the
  grasp pose — closed beside the pad. a05 never engaged — the checkpoint just
  didn't close that draw. `df5bf68`'s `--gripper-latch-near` adds a proximity
  gate (engage only within a radius of the slot's demo grasp pose) —
  **untested on the rig, a06 planned.** New open question **U-42**: fix the
  close on the rig (latch + proximity gate) or retrain with a decisive
  gripper target — see §4.

## 4. Immediate next steps (after R-84/R-85/R-86, day-3 end)

**Superseded: everything under "Immediate next steps" as it read before
14:12 (in particular, "keep the synchronous plan-then-execute line as the
default executor") is wrong — see the update in §3 above.** The historical
numbered list further down (the day-3 log of what was tried before the
compaction) is unchanged and kept for the trail; read it as history, not as
current guidance. **Superseded again at day-3 end (R-84/R-85/R-86): item 2
below (run VLA-JEPA after the transport fix) is done and did not work — see
U-41 — and item 1 (restart block B2) should run WITH the a06 proximity latch,
not the bare offset-splice line, since R-85/R-86 show the approach is solved
and the close is now the only open mechanism.**

1. **Run a06: the offset-splice line with the proximity-gated gripper
   latch** (`--gripper-latch 0.5:30:0.9 --gripper-latch-near
   slot:xNNN:0.25`, `df5bf68`, on top of R-83/R-85's line). R-86 showed a
   threshold-only latch either fires early (a04, on the pre-shape dip) or
   never fires (a05); check whether gating the engage on proximity to the
   slot's demo grasp pose fixes both failure modes and holds the close
   through the arm's drift (R-85's problem) without opening it early.
2. **Restart scored block B2 on the offset-splice line, with a06's latch
   if it passes** (`--async-inference --chunk-time-base observation
   --chunk-splice offset --splice-ramp-ticks 20 --replan-steps 8 --max-delta
   0.04`, ACT @100000, port 8767, plus the gripper latch — 16c's T6-6
   default). R-80's three rollouts on the synchronous line are no longer the
   operative evidence (U-36 answered) — re-run the frozen slot sequence
   (`t6_slot_sequence.csv`) at the protocol's `--seconds 30` before judging
   the checkpoint.
3. **VLA-JEPA: done for this window (U-41).** The transport fix (client
   resize to 224 px, two cameras) landed and cut the payload 117 kB → 17 kB,
   but RTT only dropped 0.24 s → 0.225 s — the link is latency-bound, not
   payload-bound, and still exceeds the 0.35 s chunk. VLA-JEPA needs a server
   co-located with the robot; park it until that is available and do not
   re-run it over this link expecting a different transport trim to help.
4. **Retrain ACT** on the same frozen slot sequence, for a paired comparison
   against the parent checkpoint (15 §2.2's sign-test arithmetic needs the
   pairing, not two independent scored blocks), now with two concrete
   targets instead of one: (a) image transforms (brightness/contrast/
   saturation/hue) and state noise, as before; (b) per R-86/U-42, a
   binarised/decisive gripper target (or gripper-specific temporal
   ensembling) so the checkpoint commits to a fast close instead of the
   ≈3 s ramp R-85 shows drifting the arm off the pad.
5. **Housekeeping:** open the PR for `jo/local-ebim-day-3` from the compare
   page (body in the session scratchpad `pr_day3.md`); delete `pci-sim.pem`
   on ebimHP, coordinating with the colleague before the next tunnel.

The day-3 historical log (all of it superseded by the R-81/R-82/R-83 update
in §3 — item 3 in the pre-14:12 revision of this section, "keep the
synchronous line as the default," was wrong — kept for the trail):

1. **Land U-32**: collect the agent's commit from its worktree
   (`.claude/worktrees/agent-*`, branch `worktree-agent-*`), cherry-pick onto
   `jo/local-ebim-day-3`, `make test`, a short review, rsync (marker must match
   `git rev-parse HEAD`).
2. **One rollout, same T6-6 line** (cheat sheet). Read on every `policy chunk`
   line: `splice=` (expect 0…−9), `jump=` (≪ 0.05), `lead=` (≤ 0.15); in the
   stats: `starved_ticks` ≈ 0, `leash_active_pct`, `arrival_jump_rad_max`,
   `clamped_pct`. Visual gate: no wiggle. Archive + `score_t5_trace.py` +
   one row in `runs.csv`.

   **State at 10:55: this step is done (R-72, `t6_act_105457`).** Hold is
   gone (starved 16/579), but `jump=` did not drop — the leash itself now
   produces a 0.15 rad sawtooth at chunk cadence; see 16 R-72/U-32/U-26.

   **Repeatability sample, 11:06:34 (R-72b, `t6_act_110634`, operator-run).**
   Same signature as R-72 (588 ticks, 70 chunks, `arrival_jump_rad` mean
   0.155/max 0.266, `splice_shift_mean` +1.04, `leash_active_pct` 15.0 %) —
   confirms R-72 was not a one-off before the site confounds below were
   touched.
3. **Confound 2 (wrist resolution) — DONE 2026-09-03 11:35 (R-73).**
   `enable_depth:=true` did not move colour (depth opened at 640×480, colour
   stayed 848×480); the actual fix is
   `config_file:=configs/d405_color_640x480.yml` (new file,
   `depth_module.color_profile: "640x480x30"`) on both wrist launches, because
   stock `rs_launch.py` 4.55.1 never forwards a bare `depth_module.color_profile`
   launch argument (only `config_file`). Launcher log now shows `Width: 640,
   Height: 480` on both wrists; carried into R-74 with
   `--allow-camera-shape-mismatch` dropped for good.
4. **Confound 3 (velocity cap) — DONE 2026-09-03 ~11:25.** Site raised
   `max_goal_velocity` 0.5 → 1.0 rad/s in both companion `controllers.yaml`
   copies + stack restart (both controllers logged `max_goal_velocity =
   1.000 rad/s`). Measured on R-74: hover speed p95 rose only to 0.5–0.6 rad/s
   (max 0.8) — the cap is no longer binding, and it was not gating the
   sawtooth (U-26).

   **Both confounds closed, first clean rollout done 2026-09-03 11:28:59
   (R-74, `t6_act_112859`, branch `0748bcd`, `--chunk-dump`).** Mechanism PASS,
   task ABSENT, same sawtooth as every prior run. The chunk dump is the day's
   key result: every chunk's row 0 is offset from the measured pose at
   arrival and every chunk descends on j1/j7 within its own horizon, yet the
   arm never drifts that way — a policy/executor limit cycle, not a
   confound-driven artifact. See §3 above and 16 R-74/U-33/U-34.
5. **U-33 (ensembling) is PARKED, not the next step** — measured on a stale
   worktree branch (`worktree-agent-a9bd7ec4049df7049`, commit `c00d6a4`, NOT
   merged): at the deployed 21-row horizon and 0.45 s RTT there is not enough
   chunk overlap per tick for ensembling to do anything (arrival jump halves
   but hover peak-to-peak is unchanged); it needs a longer chunk (retrain) or
   a faster round trip, neither available today. **U-35 is now CLOSED
   (R-77, R-76, R-79):** (a) the end-to-end code audit of the image/wrench
   path found no mismatch; (b) episode 163's own recorded frame 0 through the
   rig's own dummy client agreed with the demo to 0.03–0.05 rad (R-77) —
   the wire is cleared; (c) R-77 also caught a shifted/reordered live pad row
   that the overlay gate missed, it was corrected and re-rolled (R-76), and
   the chunk #1 offset was unchanged — the scene layout was ruled out too;
   (d) the live-capture + mix-and-match dummy-client probe (R-79, 12:23–12:28)
   decomposed the remaining gap into three additive causes — the head image,
   the wrist images, and the wrench/gripper state (§3, table above) — none of
   them a wire, splice, or scene bug.
   **Next, in order, now that the cause is named instead of narrowed:**
   (a) **fix the wrist exposure** (U-37): `depth_module.enable_auto_exposure:
   false` + a fixed `depth_module.exposure` in
   `configs/d405_color_640x480.yml` on both wrist launches;
   (b) **fix the head background/shadow/operator position** (U-38): match the
   recording session's head-camera lighting/shadow and keep the teleoperator
   off the right edge of frame, or crop/occlude that edge;
   (c) **address the wrench bias** (U-39): ask the colleague what payload/
   end-effector the companion carried during the original recording and
   whether the F/T sensors have been re-zeroed since, then match it or
   re-zero, and fully re-open the gripper (0.856 → 1.000) before the next
   probe;
   (d) **one synchronous rollout on the R-78 line**
   (`--no-async-inference --chunk-time-base arrival --chunk-splice index
   --replan-steps 10`) with (a)-(c) applied, to check whether the chunk #1
   row-0-vs-demo offset has shrunk toward R-77's 0.03 rad baseline — re-run
   R-79's six-send probe alongside it if time allows, to attribute any
   remaining gap;
   (e) **then repeats (n ≥ 3) of the exact R-78 line** to establish whether
   the first-full-sequence result (approach, close, transport) is repeatable
   now that the observation is closer to the recording, per U-36 — and only
   then an async control run with `--chunk-time-base arrival` to isolate
   sync/async from the arrival-time-base change, if the sequence still needs
   explaining. Do not skip straight to the scored protocol before (a)-(e).
6. Only once (a)-(e) above are done: judge the checkpoint per 15 §2 protocol
   (15 rollouts, frozen slot sequence `t6_slot_sequence.csv`, `rollouts.csv`,
   video, seed 0 for ACT), using R-78's synchronous plan-then-execute line
   if the repeats confirm it. **The live spine height reading is still
   pending** (`outputs/rig/read_spine.py`, corpus 434 mm) — the head view
   depends on it and it has not been read this session.
7. Housekeeping: open the PR for `jo/local-ebim-day-3` from the compare page
   (body in the session scratchpad `pr_day3.md`; base `dev`); the
   `pci-sim.pem` on ebimHP was due for deletion today — coordinate with the
   colleague before the next tunnel.

**Item 6 was executed, and stopped early.** R-80 (13:47–13:56) ran the block
B2 scored protocol on the R-78 synchronous line with U-37's exposure fix
applied — the (a)-(c) prerequisites above, minus U-38 (never closed, folded
into a calibration instead) and minus the (e) repeatability check on R-78
itself, which R-80 effectively answered in the negative. 3 of the planned
≥15 rollouts ran (seq 0/1/2), all ABSENT, and the operator stopped the block
rather than continue — see §3's verdict and the forward-looking items above
for what follows. Item 7 (housekeeping) is unchanged and still pending.

**Superseded 14:12–14:58 (R-81/R-82/R-83): the item-3 conclusion above (keep
the synchronous line as default) does not survive the next three hours** —
see the §3 update. An async loop with an observation-time base and the
offset splice, never tried in this window, turns out to be the line that
reaches the demo's grasp pose; the synchronous line is now demoted to a
historical alternative (16c: "do not use for ACT: R-80").

## 5. Numbers to compare the next run against

| | R-67 (index) | R-71 (nearest) | R-72 (leash) | R-74 (leash, confounds removed) | R-75 (index, confounds removed) | R-76 (index, scene corrected) | R-78 (sync plan-then-execute) | R-83 (`t6a_a02_145428`, offset splice, 120 s) | R-85 (`t6a_a03_150950`, offset splice, 120 s) | target after U-32/U-33 |
|---|---|---|---|---|---|---|---|---|---|---|
| arrival jump max / mean [rad] | 0.050 / 0.050 | 0.115 / 0.045 | 0.292 / 0.150 | 0.291 / 0.142 | 0.332 / 0.196 (vs the leashed command, not vs `nearest`'s reference) | 0.323 / 0.204 (same basis as R-75) | 0.291 / 0.128 | 0.0 / 0.0 (by construction — additive offset, not an index jump) | 0.0 / 0.0 (by construction, same splice as R-83) | ≪ 0.05 |
| splice shift mean | — | +11.1 | +0.49 (early chunks −12.9…−8.0) | −0.57 (early −2…−7.2, late +1…+2) | 0 (by construction) | 0 (by construction) | 0 (by construction) | n/a (offset splice adds an additive correction over `--splice-ramp-ticks`, not an index shift) | n/a (same as R-83) | 0…−9 |
| starved ticks / total | 0 / 587 | 413 / 587 | 16 / 579 | 0 / 587 | 0 / 588 | 0 / 588 | 0 / 369 (sync cadence, not 20 Hz) | 0 / 2375 (20 Hz, 120 s) | not logged (KeyboardInterrupt at teardown pre-empted the printed stats; CSV shows 2382 commanded rows over the full 120 s) | ≈ 0 |
| clamped_pct | 30.8 % | 4.4 % | 28.0 % | 26.4 % | 36.9 % | 40.6 % | 21.1 % | 0.1 % | 1.6 % (39/2382, from `score_t5_trace.py`) | low, and not from holding |
| leash_active_pct | n/a | n/a | 19.9 % | 10.2 % | 19.2 % | 13.6 % | 11.9 % | 34.9 % (command leads the arm by ≤0.15 rad continuously — tracking with lag, not a fault) | not logged (see starved row) | — |
| hover p2p (j6) [rad] | 0.20 | — | 0.23 | 0.14 (max across joints 0.34) | 0.25 (max across joints 0.36) | 0.30 (max across joints 0.58, j7) | n/a — arm does not hover, it executes the sequence | n/a — arm does not hover, it tracks the trajectory to the grasp pose | n/a — arm tracks to the grasp pose, closes, then hovers half-open for the last 90 s | no wiggle |
| hover speed p95 [rad/s] | ~0.5 (cap) | — | 0.4–0.5 (cap) | 0.5–0.6 (cap now 1.0, no longer binding) | not separately measured (same cap as R-74) | not separately measured (same cap as R-74/R-75) | n/a (task motion, not hover) | n/a (task motion, not hover) | n/a (task motion, then a half-open hold) | — |
| pose at t = 10 s (right j5, j6) | −1.68, 2.26 | −1.71, 2.56 | −1.65, 2.32 | −1.78, 2.18 | −1.73, 2.11 | −1.83, 1.96 | −1.86, 1.81 (still approaching at t=10; see t=20 below) | not recomputed (L2 0.34 at t=10s per R-83, below) | not recomputed (L2 0.22 at t=10s, below) | ep163 f200: −2.09, 2.60 |
| pose at t = 20 s (right j5, j6) | — | — | — | — | — | — | −2.00, 2.10 (closest yet to f200) | not recomputed — L2 min 0.08 at t=24s (below) is closer than any t=20 pose in this row | not recomputed — L2 0.14 at t=20s, min 0.09 at t=48s (below) | ep163 f200: −2.09, 2.60 |
| L2 to ep163 f200 (min) | — | — | — | 0.82 | 0.68 | 0.84 | not recomputed — see t=20 pose above (qualitatively closest) | **0.08 at t=24s** (0.34 at t=10s) — an order of magnitude closer than any run in this row | **0.09 at t=48s** (0.22 at 10s, 0.14 at 20s) — matches R-83's order of magnitude | 0 |
| chunk #1 row0 − demo f0 (right j1..j7) | — | — | — | — | +0.19,0,−0.18,+0.02,−0.02,+0.19,+0.18 | +0.17,0,−0.18,+0.01,−0.05,+0.22,+0.17 (unchanged by the scene fix) | +0.16,0,−0.18,+0.01,−0.04,+0.22,+0.16 (still unchanged) | not recomputed | not recomputed | 0 |
| gripper cmd min / task outcome | ≥0.86 (hover) | ≥0.86 (hover) | 0.919 (hover) | 0.633 (hover) | 0.985 (hover) | 0.957 (hover) | **0.20 (demo's closed value) — approach, close, transport; grasp itself pending operator report** | **0.20 (measured 0.02, closed) — best L2 to grasp pose measured yet; physical outcome (pad picked up or not) pending operator report** | **command ramps to 0.20 (measured 0.02, 25.8–28.8 s at the grasp pose), then re-opens to 0.44 and hovers half-open 90 s — pad between the fingers but the ≈3 s close ramp let the arm drift off it first (operator report, R-85)** | task-dependent |

R-72 met the "hold" and "splice direction" targets but not the jump target: the
jump is now the leash width itself (`cmd_lead_rad_max` 0.150, `leash_active_pct`
19.9 %) — see 16 U-32 for the mechanism (a chunk-cadence sawtooth, not R-68's
2 Hz clamp-jump) and U-26 for why 0.15 rad/tick worth of lead is where the
0.5 rad/s slew cap puts it. **R-74 (both site confounds removed) changes
nothing on the jump/sawtooth row**: `arrival_jump_rad` and `cmd_lead_rad_max`
are unchanged from R-72 within noise, `leash_active_pct` drops (the arm can
now follow faster) but the chunk-cadence re-anchoring itself persists — the
mechanism identified from R-74's `--chunk-dump` (16 U-33) is upstream of both
the leash and the velocity cap, so neither further site tuning nor a
different single-chunk splice was expected to move it. **R-75 (`index`
instead of `nearest`+leash) confirms exactly that**: every row above is a
different number but the same hover, and the L2-to-demo distance only
improves from 0.82 to 0.68 — still a hover, not an approach. Ensembling
(U-33) is PARKED (not enough chunk overlap at this horizon/RTT to matter).
**R-76 (scene corrected per R-77's pad-row finding) sits on the same row as
R-75 within noise** on every number above, and the chunk #1 row0-vs-demo
offset is essentially unchanged — so the scene layout is ruled out as a
cause, and R-77's wire probe (0.03–0.05 rad row-0 agreement with the demo)
rules the client→server path out too. The open lever is U-35's two
remaining suspects — live image appearance vs the recordings, or live
wrench values — via the mix-and-match dummy-client probe (§4 item 5).
**R-78 (synchronous plan-then-execute) sits in the same U-35 row as
R-75/R-76 — the offset is unchanged — but every other row changes: no
hover, `clamped_pct` drops to 21.1 % from R-76's 40.6 %, and the task
outcome flips from "hover, gripper ≥ 0.63" to "approach, close, transport"
(gripper command reaches 0.20, the demo's own closed value). This is a new
variable (executor, not scene or wire) moving the outcome while U-35 stays
open — see U-36 and §4 item 5 for what to run next before trusting it.

**R-80 (r00/r01/r02, same executor line as R-78 plus U-37's exposure fix) is
one row, not three, for this table's purposes: `arrival_jump_rad` mean
0.31/0.24/0.21 (R-78: 0.13), `leash_active_pct` 71.7/51.4/59.5 (R-78: 11.9),
gripper cmd min 0.95/0.95/0.94 (R-78: 0.20), pose at t=10s/t=20s (right j6)
1.64/1.54, 1.38/1.30, 1.32/1.30 (R-78: 1.81/2.10; target 2.60), chunk #1 row0
− demo f0 ≈ `[+0.20,0,−0.15,+0.02,−0.07,+0.15,+0.16]` in all three (unchanged
by the exposure fix, as U-35's calibration predicted it might not need to be
the whole story) — and a new sign, median row20−row0 on j6 of −0.11/−0.16/
−0.12 rad (R-78: −0.04), meaning every chunk's own horizon now pulls j6 away
from the grasp instead of toward it. **On every row that moved between R-78
and R-80, it moved in the wrong direction** — this table's "target" column
was written assuming progress toward it would be monotonic; R-80 shows it
is not.
