# Protect the Payload – Simple Digital Strategy Game (Streamlit)
# -------------------------------------------------------------
# Key behaviors:
# • PIN-gated Instructor (1557).
# • Fairness: hide hazards/results while any round is OPEN.
# • Short-lived "pulse" refresh: when Instructor opens/closes a round, clients do soft reruns
#   for ~5 seconds using st_autorefresh (no meta refresh, no visible flicker).
# • View + team/game persisted in the URL (?view=team&gid=ABC123&team=Eagles).
# • Stable widget keys ensure no loss of in-progress choices.

from __future__ import annotations
import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import time, random, string
import threading
import copy

# ---------- Require (soft) auto-reruns via st_autorefresh ----------
try:
    from streamlit import st_autorefresh
    HAS_AUTO = True
except Exception:
    HAS_AUTO = False

# ---------- Security ----------
INSTRUCTOR_PIN = "1557"

# ---------- Constants ----------
COMPONENTS = [
    ("Light Shield", 10, {"Rocky Terrain"}),
    ("Heavy Shield", 30, {"Rocky Terrain", "Heat Wave"}),
    ("Parachute", 20, {"Turbulence"}),
    ("Guidance System", 20, set()),      # dodge one hazard by name
    ("Reinforced Frame", 20, {"High Wind"}),
    ("Foam Liner", 10, set()),           # -1 total loss if there was any loss
]
HAZARDS = {"High Wind": 3, "Rocky Terrain": 3, "Heat Wave": 2, "Turbulence": 2}
DEFAULT_CONFIG = {"rounds": 5, "starting_budget": 250, "starting_payload": 20}
COMP_KEYS = [c[0] for c in COMPONENTS]
COSTS = {name: cost for name, cost, _ in COMPONENTS}
VIEWS = {"instructor": "Instructor", "team": "Team Console", "leader": "Leaderboard"}

# ---------- Shared store ----------
@st.cache_resource(show_spinner=False)
def get_store():
    # events: bump on open/close (for debugging/visibility if needed)
    # pulses: per-game "refresh until ts" so clients rerun briefly after an instructor change
    return {"games": {}, "lock": threading.Lock(), "events": {}, "pulses": {}}

# ---------- Data Models ----------
@dataclass
class TeamDecision:
    components: Dict[str, int]
    dodged_hazard: Optional[str] = None
    submitted: bool = False
    submitted_ts: Optional[float] = None
    decision_seconds: Optional[float] = None  # open->submit duration

@dataclass
class TeamState:
    name: str
    decisions: Dict[int, TeamDecision] = field(default_factory=dict)
    computed: Dict[int, Dict[str, float]] = field(default_factory=dict)

@dataclass
class GameState:
    game_id: str
    admin_pin: str
    config: Dict[str, int] = field(default_factory=lambda: DEFAULT_CONFIG.copy())
    hazards: Dict[int, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)
    round_open: Dict[int, bool] = field(default_factory=dict)
    round_open_ts: Dict[int, float] = field(default_factory=dict)
    teams: Dict[str, TeamState] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)

    def max_round(self) -> int:
        return int(self.config.get("rounds", 5))

    def ensure_rounds(self):
        rmax = self.max_round()
        for r in range(1, rmax + 1):
            self.hazards.setdefault(r, (None, None))
            self.round_open.setdefault(r, False)
            self.round_open_ts.setdefault(r, None)

# ---------- Utils ----------
def random_code(n=6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))

def latest_closed_round(gs: GameState) -> Optional[int]:
    for r in range(gs.max_round(), 0, -1):
        if gs.round_open_ts.get(r) is not None and not gs.round_open.get(r, False):
            return r
    return None

def compute_team_progress(gs: GameState, team: TeamState) -> Dict[int, Dict[str, float]]:
    rmax = gs.max_round()
    out: Dict[int, Dict[str, float]] = {}
    prev_payload = float(gs.config.get("starting_payload", 20))
    prev_budget  = float(gs.config.get("starting_budget", 150))
    total_spent  = 0.0
    inventory: Dict[str, int] = {k: 0 for k in COMP_KEYS}

    for r in range(1, rmax + 1):
        dec = team.decisions.get(r)
        haz1, haz2 = gs.hazards.get(r, (None, None))
        round_spend = 0.0
        budget_after = prev_budget
        loss1 = loss2 = 0.0
        foam_red = 0.0
        total_loss = 0.0
        payload_after = prev_payload
        inventory_after = copy.deepcopy(inventory)

        if dec and dec.submitted:
            desired = dec.components or {k: 0 for k in COMP_KEYS}
            purchases = [k for k in COMP_KEYS if inventory.get(k, 0) == 0 and desired.get(k, 0) == 1]
            round_spend = sum(COSTS[k] for k in purchases)
            budget_after = max(0.0, prev_budget - round_spend)
            total_spent_after = total_spent + round_spend
            current_inv = {k: 1 if (inventory.get(k, 0) == 1 or k in purchases) else 0 for k in COMP_KEYS}
            used = {k: 0 for k in COMP_KEYS}

            def apply_hazard(hname: Optional[str]) -> float:
                if not hname:
                    return 0.0
                base = HAZARDS[hname]
                if (
                    current_inv.get("Guidance System", 0) == 1
                    and dec.dodged_hazard and dec.dodged_hazard.strip() == hname
                    and used["Guidance System"] == 0
                ):
                    used["Guidance System"] = 1
                    return 0.0
                if hname == "Rocky Terrain":
                    if current_inv.get("Light Shield", 0) == 1 and used["Light Shield"] == 0:
                        used["Light Shield"] = 1; return 0.0
                    if current_inv.get("Heavy Shield", 0) == 1 and used["Heavy Shield"] == 0:
                        used["Heavy Shield"] = 1; return 0.0
                elif hname == "Heat Wave":
                    if current_inv.get("Heavy Shield", 0) == 1 and used["Heavy Shield"] == 0:
                        used["Heavy Shield"] = 1; return 0.0
                elif hname == "High Wind":
                    if current_inv.get("Reinforced Frame", 0) == 1 and used["Reinforced Frame"] == 0:
                        used["Reinforced Frame"] = 1; return 0.0
                elif hname == "Turbulence":
                    if current_inv.get("Parachute", 0) == 1 and used["Parachute"] == 0:
                        used["Parachute"] = 1; return 0.0
                return float(base)

            loss1 = apply_hazard(haz1)
            loss2 = apply_hazard(haz2)
            total_loss_prefoam = loss1 + loss2
            if current_inv.get("Foam Liner", 0) == 1 and total_loss_prefoam > 0 and used["Foam Liner"] == 0:
                foam_red = 1.0; used["Foam Liner"] = 1
            total_loss = max(0.0, total_loss_prefoam - foam_red)
            payload_after = max(0.0, prev_payload - total_loss)
            inventory_after = {k: (1 if current_inv.get(k, 0) == 1 and used[k] == 0 else 0) for k in COMP_KEYS}
            total_spent = total_spent_after

        out[r] = {
            "PrevPayload": prev_payload, "PayloadAfter": payload_after,
            "PrevBudget": prev_budget,   "BudgetAfter": budget_after,
            "PrevCost": total_spent,     "CurrentCost": total_spent + round_spend,
            "RoundSpend": round_spend,   "Loss1": loss1, "Loss2": loss2,
            "FoamReduction": foam_red,   "TotalLoss": total_loss,
            "InventoryAfter": inventory_after,
        }
        prev_payload = payload_after
        prev_budget  = budget_after
        inventory    = inventory_after

    team.computed = out
    return out

def prev_budget_and_inventory(gs: GameState, team: TeamState, round_index: int) -> Tuple[float, Dict[str,int]]:
    if round_index <= 1:
        return float(gs.config.get("starting_budget", 150)), {k:0 for k in COMP_KEYS}
    prog = compute_team_progress(gs, team)
    prev = prog.get(round_index - 1, {})
    budget = float(prev.get("BudgetAfter", gs.config.get("starting_budget", 150)))
    inv    = prev.get("InventoryAfter", {k:0 for k in COMP_KEYS})
    return budget, {k:int(inv.get(k,0)) for k in COMP_KEYS}

def inventory_after_round(gs: GameState, team: TeamState, upto_round: int) -> Dict[str, int]:
    if upto_round <= 0:
        return {k: 0 for k in COMP_KEYS}
    prog = compute_team_progress(gs, team)
    inv = prog.get(upto_round, {}).get("InventoryAfter")
    if isinstance(inv, dict):
        return {k: int(inv.get(k, 0)) for k in COMP_KEYS}
    return {k: 0 for k in COMP_KEYS}

def render_guides():
    st.caption("Component guide (cost → protection)")
    st.markdown("\n".join([
        f"- Light Shield (${COSTS['Light Shield']}) – protects Rocky Terrain",
        f"- Heavy Shield (${COSTS['Heavy Shield']}) – protects Rocky Terrain OR Heat Wave",
        f"- Parachute (${COSTS['Parachute']}) – protects Turbulence (no help in High Wind)",
        f"- Guidance System (${COSTS['Guidance System']}) – dodge exactly one hazard per round (type its name)",
        f"- Reinforced Frame (${COSTS['Reinforced Frame']}) – protects High Wind",
        f"- Foam Liner (${COSTS['Foam Liner']}) – reduce total round loss by 1 (min 0)",
    ]))
    st.caption("Damage guide (hazard → payload loss)")
    st.markdown("\n".join([f"- {name} – reduces payload by {HAZARDS[name]}" for name in HAZARDS]))

# ---------- URL helpers ----------
def _qp_get(name: str, default: str = "") -> str:
    try:
        val = st.query_params.get(name, default)
        if isinstance(val, list): return val[0] if val else default
        return str(val)
    except Exception:
        val = st.experimental_get_query_params().get(name, [default])
        return val[0] if isinstance(val, list) and val else default

def _qp_set(view: Optional[str] = None, gid: Optional[str] = None, team: Optional[str] = None):
    qp = {k: _qp_get(k, "") for k in ["view", "gid", "team"]}
    if view is not None: qp["view"] = str(view)
    if gid  is not None: qp["gid"]  = str(gid)
    if team is not None: qp["team"] = str(team)
    try:
        st.query_params.clear()
        for k, v in qp.items(): st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**qp)

# ---------- App UI ----------
st.set_page_config(page_title="Protect the Payload – Strategy Game", page_icon="🛡️", layout="wide")
store = get_store()

# Seed state from URL
url_view = _qp_get("view", "instructor").lower()
url_gid  = _qp_get("gid", "")
url_team = _qp_get("team", "")

if "app_view" not in st.session_state:
    st.session_state["app_view"] = {"instructor":"Instructor","team":"Team Console","leader":"Leaderboard"}.get(url_view,"Instructor")
if "team_game" not in st.session_state:
    st.session_state["team_game"] = url_gid
if "team_name" not in st.session_state:
    st.session_state["team_name"] = url_team
if "current_game" not in st.session_state:
    st.session_state["current_game"] = ""

# View picker (persisted + URL sync)
def _on_view_change():
    chosen = st.session_state["app_view"]
    inv = {v:k for k,v in VIEWS.items()}
    _qp_set(view=inv.get(chosen,"instructor"),
            gid=st.session_state.get("team_game",""),
            team=st.session_state.get("team_name",""))

mode = st.sidebar.selectbox("Choose view", list(VIEWS.values()),
                            key="app_view", on_change=_on_view_change)

st.title("🛡️ Protect the Payload – Strategy Game")
st.caption("Strategy competition for COMM401 Section 5. (c) Waqas Nawaz")

# ---------- INSTRUCTOR ----------
if mode == VIEWS["instructor"]:
    if "instructor_auth" not in st.session_state:
        st.session_state["instructor_auth"] = False

    if not st.session_state["instructor_auth"]:
        st.subheader("Instructor Login")
        pin_try = st.text_input("Enter Instructor PIN", type="password", key="instructor_pin_input")
        if st.button("Unlock"):
            if pin_try == INSTRUCTOR_PIN:
                st.session_state["instructor_auth"] = True
                st.success("Instructor view unlocked."); st.rerun()
            else:
                st.error("Incorrect PIN.")
        st.info("If you're a student, use Team Console or Leaderboard in the sidebar.")
        st.stop()
    else:
        if st.sidebar.button("🔒 Lock Instructor View"):
            st.session_state["instructor_auth"] = False; st.rerun()

    st.subheader("Instructor – Host a Game")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Create New Game", type="primary"):
            with store["lock"]:
                gid = random_code(6); pin = random_code(4)
                gs = GameState(game_id=gid, admin_pin=pin); gs.ensure_rounds()
                store["games"][gid] = gs
            st.success(f"Game created: Code **{gid}** | Admin PIN **{pin}**")
            st.session_state["current_game"] = gid

        gid_input = st.text_input("Game Code", value=st.session_state.get("current_game", ""), key="instr_gid_input")
        pin_input = st.text_input("Admin PIN", type="password", key="instr_pin_input")
        if st.button("Load Game"):
            gs = store["games"].get(gid_input)
            if gs and gs.admin_pin == pin_input:
                st.session_state["current_game"] = gid_input
                st.success("Game loaded.")
            else:
                st.error("Invalid code or PIN.")

    gid = st.session_state.get("current_game")
    if gid and gid in store["games"]:
        gs: GameState = store["games"][gid]
        st.markdown(f"### Game **{gid}**")
        cc1, cc2, cc3, cc4 = st.columns([1,1,1,1])
        with cc1:
            rounds = st.number_input("Rounds", 1, 10, value=gs.config.get("rounds", 5), key="rounds_input")
        with cc2:
            sb = st.number_input("Starting Budget", 0, 1000, value=gs.config.get("starting_budget", 150), key="sb_input")
        with cc3:
            sp = st.number_input("Starting Payload", 0, 100, value=gs.config.get("starting_payload", 20), key="sp_input")
        with cc4:
            if st.button("Save Settings"):
                with store["lock"]:
                    gs.config.update({"rounds": int(rounds), "starting_budget": int(sb), "starting_payload": int(sp)})
                    gs.ensure_rounds()
                st.success("Settings updated.")

        st.divider()
        st.markdown("#### Hazards & Round Control")
        hazard_names = list(HAZARDS.keys())
        rmax = gs.max_round()
        for r in range(1, rmax + 1):
            haz_col = st.columns([2,2,1,1,2])
            h1_default, h2_default = gs.hazards.get(r, (None, None))
            with haz_col[0]:
                h1 = st.selectbox(f"Round {r} – Hazard 1", [None] + hazard_names,
                    index=(hazard_names.index(h1_default)+1) if h1_default in hazard_names else 0, key=f"h1_{r}")
            with haz_col[1]:
                h2 = st.selectbox(f"Round {r} – Hazard 2", [None] + hazard_names,
                    index=(hazard_names.index(h2_default)+1) if h2_default in hazard_names else 0, key=f"h2_{r}")
            with haz_col[2]:
                open_flag = st.toggle("Open", value=gs.round_open.get(r, False), key=f"open_{r}")
            with haz_col[3]:
                if st.button("Save", key=f"save_{r}"):
                    with store["lock"]:
                        prev_open = gs.round_open.get(r, False)
                        gs.hazards[r] = (h1, h2)
                        gs.round_open[r] = open_flag
                        # Start a short pulse window for soft reruns (about 5 seconds)
                        pulse_until = time.time() + 5.0
                        store["pulses"][gid] = pulse_until
                        # Track transitions for analytics/debug if needed
                        if open_flag and not prev_open:
                            gs.round_open_ts[r] = time.time()
                            store["events"][gid] = store["events"].get(gid, 0) + 1
                        elif (not open_flag) and prev_open:
                            store["events"][gid] = store["events"].get(gid, 0) + 1
                    st.toast(f"Round {r} saved.")
            with haz_col[4]:
                total = len(gs.teams)
                submitted = sum(1 for t in gs.teams.values() if (t.decisions.get(r) and t.decisions[r].submitted))
                st.write(f"Submitted: **{submitted}/{total}** | Current: **{gs.hazards.get(r, (None, None))}** | Open: **{gs.round_open.get(r, False)}**")
                if total > 0 and submitted < total:
                    waiting = [t.name for t in gs.teams.values() if not (t.decisions.get(r) and t.decisions[r].submitted)]
                    st.caption("Waiting on: " + ", ".join(waiting))

        st.divider()
        st.markdown("#### Teams")
        if gs.teams:
            for tname, _ in gs.teams.items(): st.write(f"• {tname}")
        else:
            st.info("No teams yet. Ask students to join via Team Console with game code.")

        st.divider()
        if st.button("Reset Game (clear teams & progress)", type="secondary"):
            with store["lock"]:
                store["games"][gid] = GameState(game_id=gs.game_id, admin_pin=gs.admin_pin)
                store["games"][gid].ensure_rounds()
                store["events"][gid] = 0
                store["pulses"][gid] = 0
            st.warning("Game reset. Share the same code/PIN.")
    else:
        st.info("Create or load a game to configure settings & hazards.")

# ---------- TEAM CONSOLE ----------
elif mode == VIEWS["team"]:
    st.subheader("Team Console")

    gid = st.text_input("Game Code", key="team_gid_input", value=st.session_state.get("team_game",""))
    tname = st.text_input("Team Name", key="team_name_input", value=st.session_state.get("team_name",""))

    cols = st.columns([1,1,3])
    with cols[0]:
        join_clicked = st.button("Join / Load Team", type="primary")
    with cols[1]:
        if gid and tname:
            st.link_button("Copy Team Link", url=f"?view=team&gid={gid}&team={tname}")

    if join_clicked:
        if gid not in store["games"]:
            st.error("Game not found. Check the code with your instructor.")
        elif not tname:
            st.error("Please enter a team name.")
        else:
            with store["lock"]:
                gs: GameState = store["games"][gid]
                if tname not in gs.teams:
                    gs.teams[tname] = TeamState(name=tname)
                st.session_state["team_game"] = gid
                st.session_state["team_name"] = tname
            _qp_set(view="team", gid=gid, team=tname)
            st.success(f"Welcome, {tname}! You are in game {gid}.")

    gid_ss = st.session_state.get("team_game")
    tname_ss = st.session_state.get("team_name")

    if gid_ss and tname_ss and gid_ss in store["games"]:
        gs: GameState = store["games"][gid_ss]
        team: TeamState = gs.teams[tname_ss]

        # Soft rerun ONLY while a pulse is active (for ~5s after instructor Save)
        pulse_until = store.get("pulses", {}).get(gid_ss, 0)
        if HAS_AUTO and time.time() < float(pulse_until or 0):
            st_autorefresh(interval=700, key=f"pulse_{gid_ss}_{tname_ss}")

        # If st_autorefresh isn't available, show a small warning to upgrade Streamlit
        if not HAS_AUTO:
            st.warning("Note: To auto-open rounds without manual refresh on all devices, "
                       "please upgrade Streamlit to a version with `st_autorefresh`.")

        st.markdown(f"### Game **{gid_ss}** – Team **{tname_ss}**")
        gs.ensure_rounds()
        progress = compute_team_progress(gs, team)

        # Header metrics: hide results while any round is OPEN (show pre-round totals).
        rmax = gs.max_round()
        open_rounds = [r for r in range(1, rmax+1) if gs.round_open.get(r, False)]
        if open_rounds:
            r0 = min(open_rounds)
            pv = progress.get(r0, {}).get("PrevPayload", gs.config["starting_payload"])
            pb = progress.get(r0, {}).get("PrevBudget", gs.config["starting_budget"])
            display_payload, display_budget = int(pv), int(pb)
        else:
            last = rmax
            display_payload = int(progress.get(last, {}).get("PayloadAfter", gs.config["starting_payload"]))
            display_budget  = int(progress.get(last, {}).get("BudgetAfter", gs.config["starting_budget"]))

        c1, c2 = st.columns(2)
        with c1: st.metric("Payload (points)", value=display_payload)
        with c2: st.metric("Budget ($)", value=display_budget)

        st.divider()
        st.markdown("#### Rounds")
        for r in range(1, gs.max_round() + 1):
            with st.expander(f"Round {r}", expanded=(r == 1)):
                haz1, haz2 = gs.hazards.get(r, (None, None))
                open_now = gs.round_open.get(r, False)
                dec = team.decisions.get(r)
                submitted_this_round = bool(dec and dec.submitted)

                show_h1 = haz1 if (not open_now) else "Hidden"
                show_h2 = haz2 if (not open_now) else "Hidden"
                st.write(f"**Hazards:** {show_h1} & {show_h2} | **Status:** {'OPEN' if open_now else 'Closed'}")

                render_guides()

                comp_vals = progress.get(r, {})
                earlier_open = any(gs.round_open.get(k, False) for k in range(1, r))

                if not open_now:
                    if earlier_open:
                        first_open = next(k for k in range(1, r) if gs.round_open.get(k, False))
                        st.info(f"Waiting for Round {first_open} to close; results are hidden for fairness.")
                    elif comp_vals:
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Prev Payload", int(comp_vals.get("PrevPayload", 0)))
                        c2.metric("Loss", int(comp_vals.get("TotalLoss", 0)))
                        c3.metric("Payload After", int(comp_vals.get("PayloadAfter", 0)))
                        c4.metric("Spend", int(comp_vals.get("RoundSpend", 0)))
                        c5.metric("Budget After", int(comp_vals.get("BudgetAfter", 0)))

                if submitted_this_round:
                    if open_now:
                        st.info("Decision submitted. Waiting for instructor to close this round; results hidden for fairness.")
                    else:
                        st.success("Round submitted. Decisions are locked.")
                    if dec:
                        chosen = [k for k in COMP_KEYS if dec.components.get(k, 0) == 1]
                        st.markdown("**Your submitted loadout:** " + (", ".join(chosen) if chosen else "(none)"))
                        if dec.dodged_hazard:
                            st.markdown(f"**Dodged Hazard:** {dec.dodged_hazard}")

                elif not open_now:
                    st.warning("Round is CLOSED. Wait for your instructor to open it.")

                else:
                    round_prev_payload = int(progress.get(r, {}).get("PrevPayload", gs.config["starting_payload"]))
                    round_prev_budget  = int(progress.get(r, {}).get("PrevBudget",  gs.config["starting_budget"]))
                    m1, m2 = st.columns(2)
                    with m1: st.metric("Your Payload for this round", round_prev_payload)
                    with m2: st.metric("Your Budget for this round",  round_prev_budget)

                    prev_inventory = inventory_after_round(gs, team, r-1)
                    existing_flags = dec.components if dec else prev_inventory

                    with st.form(key=f"form_{tname_ss}_{r}"):
                        st.caption("Choose your components (1 = own, 0 = not owned). Only unused components carry to next round. Max 2 new purchases.")
                        cols = st.columns(3)
                        new_flags = {}
                        for i, comp_name in enumerate(COMP_KEYS):
                            with cols[i % 3]:
                                new_flags[comp_name] = st.selectbox(
                                    comp_name, options=[0, 1],
                                    index=existing_flags.get(comp_name, 0),
                                    key=f"{tname_ss}_{r}_{comp_name}"
                                )
                        dodge = st.text_input("Dodged Hazard (type exact name)", value=(dec.dodged_hazard if dec and dec.dodged_hazard else ""))
                        submitted = st.form_submit_button("Submit Round Decision")

                    if submitted:
                        if not gs.round_open.get(r, False):
                            st.warning("Round closed while you were editing. Submission ignored.")
                        else:
                            already = team.decisions.get(r)
                            if already and already.submitted:
                                st.warning(f"Round {r} is already submitted. Decisions are locked.")
                            else:
                                prev_budget_val, prev_inv = prev_budget_and_inventory(gs, team, r)
                                newly_added = [k for k in COMP_KEYS if prev_inv.get(k,0)==0 and new_flags.get(k,0)==1]
                                if len(newly_added) > 2:
                                    st.error(f"You can purchase at most 2 new components this round. You tried to buy {len(newly_added)}.")
                                else:
                                    spend = sum(COSTS[k] for k in newly_added)
                                    if spend > prev_budget_val + 1e-9:
                                        st.error(f"Insufficient budget. Costs ${int(spend)} but you have ${int(prev_budget_val)}.")
                                    else:
                                        now = time.time()
                                        opened_at = gs.round_open_ts.get(r)
                                        decision_seconds = (now - opened_at) if opened_at else None
                                        with store["lock"]:
                                            team.decisions[r] = TeamDecision(
                                                components=new_flags,
                                                dodged_hazard=dodge.strip() or None,
                                                submitted=True, submitted_ts=now,
                                                decision_seconds=decision_seconds,
                                            )
                                            compute_team_progress(gs, team)
                                        st.rerun()

# ---------- LEADERBOARD ----------
elif mode == VIEWS["leader"]:
    st.subheader("Leaderboard / Projector")
    gid = st.text_input("Game Code", value=st.session_state.get("current_game", ""), key="leader_gid_input")

    if gid and gid in store["games"]:
        # Soft rerun ONLY while a pulse is active
        pulse_until = store.get("pulses", {}).get(gid, 0)
        if HAS_AUTO and time.time() < float(pulse_until or 0):
            st_autorefresh(interval=700, key=f"pulse_leader_{gid}")
        if not HAS_AUTO:
            st.warning("Note: To auto-update instantly on close, please upgrade Streamlit to a version with `st_autorefresh`.")

        gs: GameState = store["games"][gid]
        gs.ensure_rounds()

        closed_round = latest_closed_round(gs)
        all_closed = (closed_round == gs.max_round()) if closed_round is not None else False

        rows = []
        if closed_round is not None:
            for tname, team in gs.teams.items():
                prog = compute_team_progress(gs, team)
                payload = int(prog.get(closed_round, {}).get("PayloadAfter", gs.config["starting_payload"]))
                budget  = int(prog.get(closed_round, {}).get("BudgetAfter",  gs.config["starting_budget"]))
                total_time = sum(
                    float(dec.decision_seconds)
                    for r, dec in team.decisions.items()
                    if dec and dec.submitted and dec.decision_seconds is not None and r <= closed_round
                )
                rows.append((tname, payload, budget, total_time))
            rows.sort(key=lambda x: (-x[1], -x[2], x[3]))

        if closed_round is None:
            st.info("Waiting for Round 1 to finish…")
        elif rows:
            if all_closed:
                st.success(f"Winner (after Round {closed_round}): {rows[0][0]} "
                           f"(Payload {rows[0][1]}, Budget {rows[0][2]}, Time {int(rows[0][3])}s)")
            else:
                st.info(f"Current leader (after Round {closed_round}): {rows[0][0]} "
                        f"(Payload {rows[0][1]}, Budget {rows[0][2]}, Time {int(rows[0][3])}s)")

        st.markdown("### Top 3 Teams")
        if closed_round is None:
            st.info("Waiting for Round 1 to finish…")
        else:
            top3 = rows[:3]
            st.dataframe({
                "Round":   [closed_round for _ in top3],
                "Team":    [r[0] for r in top3],
                "Payload": [r[1] for r in top3],
                "Budget":  [r[2] for r in top3],
                "Time (s)":[int(r[3]) for r in top3],
            }, use_container_width=True)

        st.markdown("### Full Standings")
        if closed_round is None:
            st.info("Waiting for Round 1 to finish…")
        else:
            st.dataframe({
                "Round":   [closed_round for _ in rows],
                "Team":    [r[0] for r in rows],
                "Payload": [r[1] for r in rows],
                "Budget":  [r[2] for r in rows],
                "Time (s)":[int(r[3]) for r in rows],
            }, use_container_width=True)

        st.markdown("### Hazards Schedule")
        hz = {}
        for r in range(1, gs.max_round()+1):
            if gs.round_open.get(r, False):
                hz[f"Round {r}"] = "Hidden + Hidden"
            else:
                h1, h2 = gs.hazards.get(r, (None, None))
                hz[f"Round {r}"] = f"{h1 or '—'} + {h2 or '—'}"
        st.dataframe(hz, use_container_width=True)

        st.caption("Tip: Keep this page open on a projector. It updates right after rounds open/close.")
    else:
        st.info("Enter a valid game code to view the live leaderboard.")
