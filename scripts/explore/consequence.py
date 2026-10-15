"""Consequence self-consistency (magnitude-free), arm A and arm B.

For every predicted chain the contact the model's OWN kinematics imply is
compared with the labels the model emitted. No reference is read.

Geometry: oriented rectangles (root length/width) at the heading of each
actor's motion between two samples (else its reported velocity), moving
linearly inside each 0.5 s interval with fixed headings; the first contact
is the lower end of the intersection of the eight separating-axis intervals,
exact under that motion model (no sampling grid). A contact strictly inside an
interval labels the previous sample, a contact on a sample labels that sample.

Evidence: a verdict rests only on the leading nodes at which the ego and EVERY
root actor have a numeric position. A contact found there stands; "no contact"
and "label without contact" are declared only when the whole chain could be
examined; otherwise the chain is `insufficient_data`.

A collision label on the LAST node of a chain: the collision rule labels the
last state before contact, so the contact lies in the following 0.5 s, which
the chain does not report. That interval is examined under TWO extrapolations
of the model's own last states, constant velocity and constant acceleration
(from the last two nodes, or from the root when the chain has one node; speed
clamped at 0). For a chain that ends early, a label supported by either is
`agree_extrapolated`; one supported by neither is `label_unsupported` (the
label is not supported by the two extrapolation assumptions, which is weaker
than a contradiction inside the reported nodes). For a six-node chain the
label requires contact on the last sample: none inside the reported nodes is
`label_without_contact` whether or not an extrapolation can be formed, and a
contact only under extrapolation (after the horizon) is
`label_beyond_horizon`, reported apart. A label on any other node
with no contact in the reported nodes is `label_without_contact`: the interval
it names was reported. An unlabelled chain that ends early and meets contact
only under extrapolation is `contact_extrapolated`, a diagnostic and not a
contradiction. A short chain whose extrapolation cannot be formed (a missing
velocity) is `insufficient_data`.

Also reported: the acceleration the model chose for the ego (first step: the
speed change while the actor still moves, a = -v0^2 / 2x once it reports
speed 0) and for the L4 treated target, the share matching the worked
example's toy rates (-2.4 ego, -3.2 target) and the published ego profile;
and, for the L4 pair, a magnitude-free direction reading (sign of the model's
own effect when it exceeds the tolerance, else "below threshold"), read on the
declared T0 axis on nuScenes and along the control target's own heading on
CARLA, the same route-normal proxy as the L4 pair measurement.

  .venv/bin/python scripts/explore/consequence.py results/api_panel results/api_panel_b --out out.csv
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.build_ground_truth import _footprint_corners

EGO_PROFILE_ACCEL = {"do_ego_panic_brake": -9.0, "do_ego_emergency_brake": -8.0, "do_ego_comfortable_brake": -3.0,
                     "do_ego_decelerate_coast": -1.0, "do_ego_no_action": 0.0, "do_ego_swerve_left": 0.0,
                     "do_ego_swerve_right": 0.0, "do_ego_accelerate_pass": 3.0, "do_ego_accelerate_escape": 3.5}
EXAMPLE_EGO, EXAMPLE_TARGET = -2.4, -3.2
PENETRATION_M = 0.25    # deeper overlap than this at the labelled node contradicts "at or before contact"
STOP_MPS = 0.5
TOL_M, TOL_MPS = 1.0, 1.0
LONGITUDINAL = {"npc_sudden_brake"}
LATERAL = {"npc_cut_in_left", "npc_cut_in_right", "pedestrian_enters_road", "cyclist_enters_road"}


def num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


# ── chains → per-slot worlds ──────────────────────────────────────────────

def states_from_chain(chain, root_by_id):
    """One dict per chain SLOT (a malformed node is an empty slot, so indices
    never shift): {actor id: state with x, y, vx, vy, speed and root dims}."""
    out = []
    for node in chain if isinstance(chain, list) else []:
        world = {}
        actors = node.get("actors_update") if isinstance(node, dict) else None
        for a in actors if isinstance(actors, list) else []:
            if not isinstance(a, dict) or a.get("id") not in root_by_id:
                continue
            x, y = num(a.get("x")), num(a.get("y"))
            if x is None or y is None:
                continue
            base = root_by_id[a["id"]]
            st = {"id": a["id"], "x": x, "y": y, "vx": num(a.get("vx")), "vy": num(a.get("vy")), "speed": num(a.get("speed"))}
            for k in ("length", "width", "role_type", "entity_type"):
                if k in base:
                    st[k] = base[k]
            world[a["id"]] = st
        out.append(world)
    return out


def label_terminal(chain):
    """(terminal label, slot index of the last object node) — slots as in states_from_chain."""
    if not isinstance(chain, list):
        return None, None
    idx = max((i for i, n in enumerate(chain) if isinstance(n, dict)), default=None)
    if idx is None:
        return None, None
    ev = chain[idx].get("event_type")
    return (ev if ev in ("collision", "safe_stop") else None), idx


def evidence_prefix(states, root_by_id, ego_id):
    """Leading slots at which the ego and every root actor have a position."""
    n = 0
    for w in states:
        if ego_id in w and all(aid in w for aid in root_by_id):
            n += 1
        else:
            break
    return n


# ── geometry ──────────────────────────────────────────────────────────────

def heading_of(prev_state, state):
    """Heading (deg) from the displacement since the previous sample, else the
    reported velocity, else 0 (along +x). One rule for every geometric check."""
    if prev_state is not None:
        dx, dy = state["x"] - prev_state["x"], state["y"] - prev_state["y"]
        if math.hypot(dx, dy) > 0.1:
            return math.degrees(math.atan2(dy, dx))
    vx, vy = state.get("vx"), state.get("vy")
    if vx is not None and vy is not None and math.hypot(vx, vy) > 0.1:
        return math.degrees(math.atan2(vy, vx))
    return 0.0


def _box(state, x, y, heading):
    return _footprint_corners(x, y, heading, state.get("length", 4.0), state.get("width", 2.0))


def _axes(poly):
    out = []
    for i in range(4):
        ex, ey = poly[(i + 1) % 4][0] - poly[i][0], poly[(i + 1) % 4][1] - poly[i][1]
        n = math.hypot(ex, ey) or 1.0
        out.append((-ey / n, ex / n))
    return out


def penetration(a, b):
    """Overlap depth of two oriented rectangles (>= 0 when they overlap or touch, -1 when separated)."""
    depth = float("inf")
    for nx, ny in _axes(a) + _axes(b):
        pa = [px * nx + py * ny for px, py in a]; pb = [px * nx + py * ny for px, py in b]
        overlap = min(max(pa), max(pb)) - max(min(pa), min(pb))
        if overlap < -1e-9:
            return -1.0
        depth = min(depth, max(overlap, 0.0))
    return depth


def first_contact_fraction(e0, e1, o0, o1, he, ho):
    """Exact first contact fraction f in [0, 1] of two rectangles moving
    linearly with fixed headings: on every separating axis the projected
    centre distance is linear in f and the half-extents are constant, so each
    axis admits contact on one f-interval; the first contact is the lower end
    of the intersection of the eight intervals. None when they never meet."""
    box_e, box_o = _box(e1, 0.0, 0.0, he), _box(o1, 0.0, 0.0, ho)      # extents only, centred at the origin
    lo, hi = 0.0, 1.0
    for nx, ny in _axes(box_e) + _axes(box_o):
        half = (max(px * nx + py * ny for px, py in box_e) - min(px * nx + py * ny for px, py in box_e)) / 2 \
             + (max(px * nx + py * ny for px, py in box_o) - min(px * nx + py * ny for px, py in box_o)) / 2
        d0 = (o0["x"] - e0["x"]) * nx + (o0["y"] - e0["y"]) * ny
        d1 = (o1["x"] - e1["x"]) * nx + (o1["y"] - e1["y"]) * ny
        # |d0 + (d1 - d0) f| <= half
        slope = d1 - d0
        if abs(slope) < 1e-12:
            if abs(d0) > half + 1e-9:
                return None
            continue
        f_a, f_b = (-half - d0) / slope, (half - d0) / slope
        lo, hi = max(lo, min(f_a, f_b)), min(hi, max(f_a, f_b))
        if lo > hi + 1e-12:
            return None
    return max(lo, 0.0)


# ── contact scan ─────────────────────────────────────────────────────────

def _extrapolated(states, mode, root_by_id=None):
    """A virtual world 0.5 s after the last slot: constant velocity, or constant
    acceleration from the last two slots (the root and the first slot for a
    one-node chain; per actor, speed clamped at 0). None when a velocity is
    missing, so the two assumptions are always both formed or neither."""
    last = states[-1]
    prev = states[-2] if len(states) > 1 else {aid: {"vx": num(a.get("vx")), "vy": num(a.get("vy"))}
                                                for aid, a in (root_by_id or {}).items()}
    virt = {}
    for aid, st in last.items():
        vx, vy = st.get("vx"), st.get("vy")
        if vx is None or vy is None:
            return None
        if mode == "accel":
            if aid not in prev or prev[aid].get("vx") is None or prev[aid].get("vy") is None:
                return None
            ax, ay = (vx - prev[aid]["vx"]) / 0.5, (vy - prev[aid]["vy"]) / 0.5
            speed = math.hypot(vx, vy)
            if speed > 1e-9:
                along = (vx * ax + vy * ay) / speed                    # signed acceleration along the motion
                t_stop = speed / -along if along < 0 else None
                t = min(0.5, t_stop) if t_stop is not None else 0.5
                s = speed * t + 0.5 * along * t * t
                virt[aid] = dict(st, x=st["x"] + vx / speed * s, y=st["y"] + vy / speed * s)
                continue
        virt[aid] = dict(st, x=st["x"] + 0.5 * vx, y=st["y"] + 0.5 * vy)
    return virt


def scan(seq, ego_id):
    """(interval index i, contact fraction, partners) of the earliest contact in a
    sequence of worlds (seq[0] is the root), or (None, None, [])."""
    for i in range(1, len(seq)):
        w0, w1 = seq[i - 1], seq[i]
        if ego_id not in w1 or ego_id not in w0:
            continue
        e0, e1 = w0[ego_id], w1[ego_id]
        he = heading_of(e0, e1)
        best, partners = None, []
        for oid, o1 in w1.items():
            if oid == ego_id or oid not in w0:
                continue
            o0 = w0[oid]
            f = first_contact_fraction(e0, e1, o0, o1, he, heading_of(o0, o1))
            if f is None:
                continue
            if best is None or f < best - 1e-9:
                best, partners = f, [oid]
            elif abs(f - best) <= 1e-9:
                partners.append(oid)
        if best is not None:
            return i, best, sorted(partners)
    return None, None, []


def contact_scan(states, root_by_id, ego_id, extrapolate):
    """Contact implied by the reported nodes, and (for a chain that ends
    early) by extrapolating its last states. Returns a dict with
    reported: (node index of the last node at or before contact, or "first_interval"
    when the contact precedes the first sample), partners, evidence
    ("complete"/"partial"), and extrapolated: {"velocity": bool, "accel": bool}."""
    prefix = evidence_prefix(states, root_by_id, ego_id)
    root = dict((k, dict(v)) for k, v in root_by_id.items())
    seq = [root] + [dict(w) for w in states[:prefix]]
    complete = prefix == len(states)
    i, f, partners = scan(seq, ego_id)
    out = {"node": None, "first_interval": False, "partners": partners, "evidence": "complete" if complete else "partial",
           "extrapolated": {}}
    if i is not None:
        node = i - 1 if f >= 1 - 1e-9 else i - 2
        if node < 0:
            out["first_interval"] = True; out["node"] = 0          # contact before the first sample: no valid node exists
        else:
            out["node"] = node
        return out
    if extrapolate and complete and states:
        for mode in ("velocity", "accel"):
            virt = _extrapolated(states[:prefix], mode, root_by_id)
            if virt is None:
                out["extrapolated"][mode] = None; continue
            j, g, p = scan([seq[-1], virt], ego_id)
            out["extrapolated"][mode] = j is not None
            if j is not None and not out["partners"]:
                out["partners"] = p
    return out


def penetration_at(states, j, root_by_id, ego_id):
    """Deepest body overlap between the ego and any actor at slot j (headings
    from the motion into slot j), or None when an actor is missing there."""
    w = states[j]
    if ego_id not in w or any(aid not in w for aid in root_by_id):
        return None, None
    prev = states[j - 1] if j > 0 else root_by_id
    ego = w[ego_id]; he = heading_of(prev.get(ego_id), ego)
    worst, who = 0.0, None
    for oid, o in w.items():
        if oid == ego_id:
            continue
        d = penetration(_box(ego, ego["x"], ego["y"], he), _box(o, o["x"], o["y"], heading_of(prev.get(oid), o)))
        if d > worst:
            worst, who = d, oid
    return worst, who


# ── chosen magnitudes ────────────────────────────────────────────────────

def accel_estimate(states, root_by_id, aid):
    """The acceleration the model gave `aid` over the first step: from the
    speed change while the actor still reports motion at depth 1 (any speed
    above zero; the 0.5 m/s stop label is not a zero), from the displacement
    (a = -v0^2 / 2x) once it reports speed 0, None otherwise."""
    root = root_by_id.get(aid)
    if not root or not states or aid not in states[0]:
        return None
    v0 = num(root.get("speed")); s1 = states[0][aid]; v1 = s1.get("speed")
    if v0 is None or v1 is None:
        return None
    if v1 > 1e-9:
        return (v1 - v0) / 0.5
    x = math.hypot(s1["x"] - (num(root.get("x")) or 0.0), s1["y"] - (num(root.get("y")) or 0.0))
    if v0 > STOP_MPS and x > 1e-6:
        return -(v0 * v0) / (2 * x)
    return None


def _lateral_normal(ctrl_states, k, target, root_by_id):
    """The +y-side normal of the control target's own heading at slot k
    (velocity there, else the displacement into it, else the root velocity),
    the route-normal proxy the L4 pair measurement uses on CARLA; the T0 +y
    axis when no heading can be read."""
    st = ctrl_states[k][target]
    prev = ctrl_states[k - 1].get(target) if k > 0 else root_by_id.get(target)
    candidates = [(st.get("vx"), st.get("vy"))]
    if prev is not None and prev.get("x") is not None:
        candidates.append((st["x"] - prev["x"], st["y"] - prev["y"]))
    root = root_by_id.get(target) or {}
    candidates.append((num(root.get("vx")), num(root.get("vy"))))
    for vx, vy in candidates:
        if vx is not None and vy is not None and math.hypot(vx, vy) > 0.1:
            n = math.hypot(vx, vy)
            return (-vy / n, vx / n)
    return (0.0, 1.0)


def own_effect_direction(ctrl_states, trt_states, target, itype, root_by_id, split="nuscenes"):
    """Magnitude-free L4 direction from the model's OWN pair: the target's
    treated-minus-control effect at the last common slot; 'below_threshold'
    when it stays under the tolerance, else whether its sign is the declared
    one. A lateral effect is read on the declared T0 +y axis on nuScenes and
    along the control target's own heading normal on CARLA (the pair
    measurement's route-normal proxy; the target's T0 side decides which
    way the ego lane lies)."""
    k = None
    for i in range(min(len(ctrl_states), len(trt_states))):
        if target in ctrl_states[i] and target in trt_states[i]:
            k = i
    if k is None:
        return "undefined"
    c, t = ctrl_states[k][target], trt_states[k][target]
    if itype in LONGITUDINAL:
        if c.get("speed") is None or t.get("speed") is None:
            return "undefined"
        eff = t["speed"] - c["speed"]
        if abs(eff) < TOL_MPS:
            return "below_threshold"
        return "sign_ok" if eff < 0 else "sign_wrong"
    if itype in LATERAL:
        y0 = num((root_by_id.get(target) or {}).get("y"))
        if y0 is None or abs(y0) < 1e-6:
            return "undefined"
        nx, ny = _lateral_normal(ctrl_states, k, target, root_by_id) if split == "carla" else (0.0, 1.0)
        eff = nx * (t["x"] - c["x"]) + ny * (t["y"] - c["y"])           # + = towards the +y side
        if abs(eff) < TOL_M:
            return "below_threshold"
        return "sign_ok" if (eff > 0) == (y0 < 0) else "sign_wrong"   # towards the ego lane from its side
    return "undefined"


# ── per-cell analysis ────────────────────────────────────────────────────

def analyse_cell(cell_dir: Path, inputs: Path, config: str, condition: str, split: str, arm: str):
    rows = []
    for l3_path in sorted(cell_dir.glob("scenario_*/l3_pre.json")):
        scen = l3_path.parent.name
        state_path = inputs / scen / "observed_state.json"
        if not state_path.exists():
            continue
        root = json.load(open(state_path)); root_actors = [a for a in root.get("actors", []) if isinstance(a, dict) and "id" in a]
        root_by_id = {a["id"]: a for a in root_actors}
        ego_id = next((a["id"] for a in root_actors if a.get("role_type") == "ego_agent" or a["id"] == "ego"), "ego")
        try:
            l3 = json.load(open(l3_path)); l4 = json.load(open(l3_path.parent / "l4_pre.json"))
        except Exception:
            continue
        items = []
        for b in l3.get("predicted_branches") or []:
            if isinstance(b, dict):
                iv = b.get("intervention") if isinstance(b.get("intervention"), dict) else {}
                items.append((iv.get("intervention_type") or str(b.get("branch_id")), "L3", b.get("chain"), None, None))
        for c in l4.get("predicted_counterfactuals") or []:
            if isinstance(c, dict):
                iv = c.get("intervention") if isinstance(c.get("intervention"), dict) else {}
                tid, itype = iv.get("target_actor_id"), iv.get("intervention_type")
                items.append((f"L4_control:{itype}", "L4c", c.get("control_chain"), tid, itype))
                items.append((f"L4_treated:{itype}", "L4t", c.get("treated_chain") or c.get("chain"), tid, itype))
                items.append((f"L4_pair:{itype}", "L4pair", (c.get("control_chain"), c.get("treated_chain") or c.get("chain")), tid, itype))
        for action, kind, chain, tid, itype in items:
            base = dict(arm=arm, config=config, split=split, condition=condition, scenario=scen, action=action, kind=kind)
            if kind == "L4pair":
                cs, ts = states_from_chain(chain[0], root_by_id), states_from_chain(chain[1], root_by_id)
                rows.append(dict(base, category=own_effect_direction(cs, ts, tid, itype, root_by_id, split) if tid else "undefined",
                                 label=None, label_idx=None, implied_idx=None, impossible=None, penetration=None, partner=None,
                                 n_nodes=None, ego_accel=None, target_accel=None, evidence=None))
                continue
            states = states_from_chain(chain, root_by_id)
            ego_accel = accel_estimate(states, root_by_id, ego_id) if kind == "L3" else None
            target_accel = accel_estimate(states, root_by_id, tid) if kind == "L4t" and tid else None
            ev_l, idx_l = label_terminal(chain)
            common = dict(base, label=ev_l, label_idx=idx_l, n_nodes=len(states), ego_accel=ego_accel, target_accel=target_accel)
            if not states:
                rows.append(dict(common, category="empty_chain", implied_idx=None, impossible=None, penetration=None, partner=None, evidence=None))
                continue
            if evidence_prefix(states, root_by_id, ego_id) == 0:
                rows.append(dict(common, category="insufficient_data", implied_idx=None, impossible=None, penetration=None, partner=None, evidence="none"))
                continue
            short = len(states) < 6
            last_label = ev_l == "collision" and idx_l == len(states) - 1
            # The 0.5 s after the last node is examined when the chain ends
            # early, and when a collision label sits on the last node (its
            # contact is in that interval; after the horizon for six nodes).
            res = contact_scan(states, root_by_id, ego_id, extrapolate=short or last_label)
            pen, who = penetration_at(states, idx_l, root_by_id, ego_id) if ev_l == "collision" and idx_l is not None and idx_l < len(states) else (None, None)
            reported = res["node"] is not None
            ext = res["extrapolated"]
            ext_formed = bool(ext) and all(v is not None for v in ext.values())
            ext_hit = ext_formed and any(ext.values())
            complete = res["evidence"] == "complete"
            if res["first_interval"]:
                cat = "first_interval_contact"                 # no node can satisfy the contract; not judged
                pen = None
            elif ev_l == "collision" and reported:
                cat = "agree" if res["node"] == idx_l else ("label_late" if idx_l > res["node"] else "label_early")
            elif ev_l == "collision" and not complete:
                cat = "insufficient_data"
            elif last_label and short and not ext_formed:
                cat = "insufficient_data"                      # the interval the label names cannot be examined
            elif last_label and ext_hit:
                cat = "agree_extrapolated" if short else "label_beyond_horizon"
            elif last_label and short:
                cat = "label_unsupported"                      # neither extrapolation reaches contact
            elif ev_l == "collision":
                cat = "label_without_contact"                  # a complete chain shows no contact on the labelled sample
            elif reported:
                cat = "contact_unlabelled"
            elif not complete:
                cat = "insufficient_data"
            elif short and not ext_formed:
                cat = "insufficient_data"
            elif ext_hit:
                cat = "contact_extrapolated"
            else:
                cat = "no_collision_either"
            rows.append(dict(common, category=cat, implied_idx=res["node"], impossible=(int(pen > PENETRATION_M) if pen is not None else None),
                             penetration=(round(pen, 3) if pen is not None else None), partner=(",".join(res["partners"]) or who),
                             evidence=res["evidence"]))
    return rows


def cell_inputs(base: Path, split: str) -> Path:
    """inputs/<split> of a cell's root. A root pulled from the pod without
    inputs/ (rsync --exclude '/inputs/') gets it back from the dataset
    directories, checked row by row against its frozen.json
    (reparse_cells.split_inputs); a root with neither stops discovery, so no
    finished cell is left out of a report unnoticed."""
    inputs = base / "inputs" / split
    if inputs.is_dir():
        return inputs
    frozen = base / "frozen.json"
    if not frozen.exists():
        raise FileNotFoundError(f"{base}: cells for {split} but neither inputs/{split} nor frozen.json")
    from scripts.reparse_cells import split_inputs
    return split_inputs(base, split, json.loads(frozen.read_text()))


def discover(root: Path):
    for cell in sorted(root.glob("**/cells/*/*")):
        if not cell.is_dir() or not any(cell.glob("scenario_*/l3_pre.json")):
            continue
        split = cell.parent.name; base = cell.parent.parent.parent
        inputs = cell_inputs(base, split)
        name = cell.name
        if "baselines" in cell.parts:
            config, cond = name, "baseline"
        else:
            for c in ("video_history", "history_json", "video"):
                if name.endswith("_" + c):
                    config, cond = name[: -len(c) - 1], c; break
            else:
                config, cond = name, "?"
        arm = "A"
        mf = next(cell.glob("scenario_*/run_manifest.json"), None)
        if mf is not None:
            try:
                variant = json.load(open(mf)).get("prompt_variant", "full")
                arm = {"full": "A", "recipe_withheld": "B", "ego_profiles_withheld": "C"}.get(variant, "?")
            except Exception:
                pass
        yield cell, inputs, config, cond, split, arm


def share(rows, pred, denom_pred):
    d = [r for r in rows if denom_pred(r)]
    return (sum(1 for r in d if pred(r)) / len(d) if d else float("nan")), len(d)


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["split"], r["config"], r["condition"])].append(r)
    print("\nshares with their own denominators: chains = every chain; evaluable = a verdict could be reached;")
    print("lbl-no = label without contact in the reported nodes; lbl-ux = label on the last node of a short chain, neither")
    print("extrapolation reaches contact; lbl-bh = six-node chain, contact only after the horizon; imposs = collision labels")
    print("judged at a complete node; ex-copy/pub-copy = ego branches whose acceleration was estimable; tgt-ex = treated targets")
    print("%-3s %-26s %-9s %-13s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %8s %8s %7s" % (
        "arm", "config", "split", "cond", "chains", "n/a", "agree", "agrX", "late", "lbl-no", "lbl-ux", "lbl-bh", "C-unl", "C-ext", "1stI", "imposs", "ex-copy", "pub-copy", "tgt-ex"))
    for key in sorted(groups):
        g = [r for r in groups[key] if r["kind"] != "L4pair"]
        undefined = {"insufficient_data", "empty_chain", "first_interval_contact"}
        ev = [r for r in g if r["category"] not in undefined]
        c = Counter(r["category"] for r in g)
        n = len(ev)
        imp, n_imp = share(g, lambda r: r["impossible"] == 1, lambda r: r["impossible"] is not None)
        l3 = [r for r in g if r["kind"] == "L3" and r["ego_accel"] is not None]
        ex = sum(1 for r in l3 if abs(r["ego_accel"] - EXAMPLE_EGO) <= 0.05) / len(l3) if l3 else float("nan")
        prof = [r for r in l3 if r["action"] in EGO_PROFILE_ACCEL]
        # published-value reproduction: the branch's ego acceleration equals ITS action's published profile
        pub = sum(1 for r in prof if abs(r["ego_accel"] - EGO_PROFILE_ACCEL[r["action"]]) <= 0.05) / len(prof) if prof else float("nan")
        l4t = [r for r in g if r["kind"] == "L4t" and r["target_accel"] is not None]
        tgt = sum(1 for r in l4t if abs(r["target_accel"] - EXAMPLE_TARGET) <= 0.05) / len(l4t) if l4t else float("nan")
        def s(cat):
            return c[cat] / n if n else float("nan")
        print("%-3s %-26s %-9s %-13s %6d %6d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6d %6.3f %8.3f %8.3f %7.3f" % (
            key[0], key[2], key[1], key[3], len(g), c["insufficient_data"] + c["empty_chain"], s("agree"), s("agree_extrapolated"),
            s("label_late") + s("label_early"), s("label_without_contact"), s("label_unsupported"), s("label_beyond_horizon"),
            s("contact_unlabelled"), s("contact_extrapolated"),
            c["first_interval_contact"], imp, ex, pub, tgt) + f"   (n_eval={n}, n_imposs={n_imp}, n_accel={len(l3)}, n_target_accel={len(l4t)})")
    print("\nL4 own-effect direction (magnitude-free): shares over pairs")
    for key in sorted(groups):
        p = [r for r in groups[key] if r["kind"] == "L4pair"]
        if not p:
            continue
        c = Counter(r["category"] for r in p); n = len(p)
        print("%-3s %-26s %-9s %-13s pairs=%4d sign_ok=%.3f sign_wrong=%.3f below_threshold=%.3f undefined=%.3f" % (
            key[0], key[2], key[1], key[3], n, c["sign_ok"] / n, c["sign_wrong"] / n, c["below_threshold"] / n, c["undefined"] / n))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("roots", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="configuration names to include")
    ap.add_argument("--conditions", nargs="*", help="conditions to include (default all; baselines always)")
    a = ap.parse_args(); rows = []
    for r in a.roots:
        for cell, inputs, config, cond, split, arm in discover(Path(r)):
            if a.only and config not in a.only:
                continue
            if a.conditions and cond not in a.conditions and cond != "baseline":
                continue
            new = analyse_cell(cell, inputs, config, cond, split, arm); rows += new
            print(f"{arm} {config:28s} {split:8s} {cond:13s} chains={len(new)}", flush=True)
    if not rows:
        print("no chains found"); return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("arm", "config", "split", "condition", "scenario", "action", "kind", "category"), k))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
