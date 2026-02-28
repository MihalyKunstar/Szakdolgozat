#!/usr/bin/env python3
"""
Single-file Mesa prototype: 1 ward hospital contact network generation (NO infection dynamics).

Outputs:
- outputs/visit_log.csv
- outputs/aggregated_edges.csv
- outputs/run_summary.csv
- outputs/figures/network.png
- outputs/figures/timeseries.png
- outputs/figures/degree_hist.png
"""

# =========================================================
# 1) Imports + constants
# =========================================================

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mesa import Agent, Model


DEFAULT_SEED = 42
DEFAULT_DT_MIN = 5
MINUTES_PER_DAY = 24 * 60
TICKS_PER_DAY = MINUTES_PER_DAY // DEFAULT_DT_MIN  # 288

N_PATIENTS = 50
N_NURSES = 10
N_DOCTORS = 5
N_ROOMS = 13
DURATION_MIN_DEFAULT = 5

# Room capacity specification:
# room_0..room_1 => 2 beds
# room_2..room_11 => 4 beds
# room_12 => 6 beds
ROOM_CAPACITY_SPEC = [2, 2] + [4] * 10 + [6]

# Time blocks in minutes [start, end), local day
DOCTOR_BLOCKS = [(7 * 60, 8 * 60), (15 * 60, 16 * 60)]
NURSE_ROUNDS_BLOCKS = [(6 * 60, 7 * 60), (16 * 60, 17 * 60)]
FEEDING_BLOCKS = [(8 * 60, 9 * 60), (12 * 60, 13 * 60), (18 * 60, 19 * 60)]
AD_HOC_BLOCK = (9 * 60, 15 * 60)
HANDOVER_BLOCKS = [(6 * 60 + 55, 7 * 60 + 5), (15 * 60 + 55, 16 * 60 + 5)]


# =========================================================
# 2) Parameters / configuration
# =========================================================

@dataclass
class SimConfig:
    seed: int = DEFAULT_SEED
    run_id: str = ""

    dt_min: int = DEFAULT_DT_MIN
    ticks_per_day: int = TICKS_PER_DAY

    n_patients: int = N_PATIENTS
    n_nurses: int = N_NURSES
    n_doctors: int = N_DOCTORS
    n_rooms: int = N_ROOMS

    feeding_coverage_min: float = 0.60
    feeding_coverage_max: float = 0.80

    p_ad_hoc_tick: float = 0.20
    ad_hoc_max_events_per_tick: int = 2

    p_roommate_event_per_room_per_hour: float = 0.10

    nurse_station_random_ticks_per_day: int = 10

    output_dir: str = "outputs"


# =========================================================
# 3) Time helper functions (tick -> HH:MM, block checks)
# =========================================================

def tick_to_time_min(tick: int, dt_min: int) -> int:
    return tick * dt_min


def minutes_to_hhmm(total_minutes: int) -> str:
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def in_any_block(time_min: int, blocks: list[tuple[int, int]]) -> bool:
    return any(start <= time_min < end for start, end in blocks)


def in_block(time_min: int, block: tuple[int, int]) -> bool:
    start, end = block
    return start <= time_min < end


def block_index(time_min: int, blocks: list[tuple[int, int]]) -> int | None:
    for idx, (start, end) in enumerate(blocks):
        if start <= time_min < end:
            return idx
    return None


# =========================================================
# 4) Agent classes
# =========================================================

class BaseHospitalAgent(Agent):
    def __init__(self, model: Model, unique_id: str, agent_type: str):
        super().__init__(model)
        self.unique_id = unique_id
        self.agent_type = agent_type


class PatientAgent(BaseHospitalAgent):
    def __init__(self, model: Model, unique_id: str, room_id: str):
        super().__init__(model, unique_id, "patient")
        self.room_id = room_id


class NurseAgent(BaseHospitalAgent):
    def __init__(self, model: Model, unique_id: str):
        super().__init__(model, unique_id, "nurse")
        self.caseload_rooms: list[str] = []
        self.is_active_feeder: bool = False


class DoctorAgent(BaseHospitalAgent):
    def __init__(self, model: Model, unique_id: str):
        super().__init__(model, unique_id, "doctor")
        self.panel_patients: list[str] = []


# =========================================================
# 5) Mesa Model class
# =========================================================

class HospitalContactModel(Model):
    def __init__(self, config: SimConfig):
        super().__init__()
        self.config = config
        self.rng = random.Random(config.seed)

        # Core maps
        self.room_capacity_map: dict[str, int] = self._build_room_capacity_map()
        self.room_occupants: dict[str, list[str]] = {rid: [] for rid in self.room_capacity_map}

        # Agent registries
        self.patients: list[PatientAgent] = []
        self.nurses: list[NurseAgent] = []
        self.doctors: list[DoctorAgent] = []
        self.agent_index: dict[str, BaseHospitalAgent] = {}

        # Simulation logs
        self.visit_events: list[dict] = []

        # Daily bookkeeping
        self.current_tick = 0
        self.current_time_min = 0

        # Prevent duplicate doctor-patient per block
        self._doctor_block_seen_pairs: dict[int, set[tuple[str, str]]] = {
            0: set(),
            1: set(),
        }

        # Track one roommate event per room/hour max (if event generated)
        self._room_hour_triggered: set[tuple[str, int]] = set()

        # Random daytime nurse station ticks
        self._random_nurse_station_ticks = self._sample_random_nurse_station_ticks()

        # Build agents and assignments
        self._init_agents()
        self._assign_patients_to_rooms_deterministic()
        self._assign_nurse_room_caseloads()
        self._assign_doctor_panels()
        self._assign_daily_feeders()

    def _build_room_capacity_map(self) -> dict[str, int]:
        assert len(ROOM_CAPACITY_SPEC) == self.config.n_rooms
        return {f"room_{i}": cap for i, cap in enumerate(ROOM_CAPACITY_SPEC)}

    def _init_agents(self):
        for i in range(self.config.n_patients):
            pid = f"patient_{i}"
            p = PatientAgent(self, pid, room_id="")
            self.patients.append(p)
            self.agent_index[pid] = p

        for i in range(self.config.n_nurses):
            nid = f"nurse_{i}"
            n = NurseAgent(self, nid)
            self.nurses.append(n)
            self.agent_index[nid] = n

        for i in range(self.config.n_doctors):
            did = f"doctor_{i}"
            d = DoctorAgent(self, did)
            self.doctors.append(d)
            self.agent_index[did] = d

    def _assign_patients_to_rooms_deterministic(self):
        patient_iter = iter(self.patients)
        for room_id, capacity in self.room_capacity_map.items():
            for _ in range(capacity):
                patient = next(patient_iter)
                patient.room_id = room_id
                self.room_occupants[room_id].append(patient.unique_id)

        total_capacity = sum(self.room_capacity_map.values())
        assert total_capacity == self.config.n_patients, "Room capacity must exactly match patient count"

    def _assign_nurse_room_caseloads(self):
        room_ids = sorted(self.room_capacity_map.keys(), key=lambda x: int(x.split("_")[1]))
        for i, room_id in enumerate(room_ids):
            nurse = self.nurses[i % len(self.nurses)]
            nurse.caseload_rooms.append(room_id)

    def _assign_doctor_panels(self):
        patient_ids = [p.unique_id for p in self.patients]
        for i, pid in enumerate(patient_ids):
            doctor = self.doctors[i % len(self.doctors)]
            doctor.panel_patients.append(pid)

    def _assign_daily_feeders(self):
        feeder_indices = self.rng.sample(range(len(self.nurses)), k=2)
        for i, nurse in enumerate(self.nurses):
            nurse.is_active_feeder = i in feeder_indices

    def _sample_random_nurse_station_ticks(self) -> set[int]:
        daytime_tick_start = AD_HOC_BLOCK[0] // self.config.dt_min
        daytime_tick_end = AD_HOC_BLOCK[1] // self.config.dt_min
        all_daytime_ticks = list(range(daytime_tick_start, daytime_tick_end))

        k = min(self.config.nurse_station_random_ticks_per_day, len(all_daytime_ticks))
        return set(self.rng.sample(all_daytime_ticks, k=k))

    def _record_event(
        self,
        tick: int,
        actor_id: str,
        target_id: str,
        room_id: str,
        event_type: str,
        duration_min: int = DURATION_MIN_DEFAULT,
    ):
        actor = self.agent_index[actor_id]
        target = self.agent_index[target_id]
        time_min = tick_to_time_min(tick, self.config.dt_min)

        event = {
            "run_id": self.config.run_id,
            "tick": tick,
            "time_min": time_min,
            "time_str": minutes_to_hhmm(time_min),
            "actor_id": actor_id,
            "actor_type": actor.agent_type,
            "target_id": target_id,
            "target_type": target.agent_type,
            "room_id": room_id,
            "event_type": event_type,
            "duration_min": duration_min,
        }
        self.visit_events.append(event)

    # =========================================================
    # 6) Contact generation logic
    # =========================================================

    def _generate_doctor_rounds(self, tick: int, time_min: int):
        bidx = block_index(time_min, DOCTOR_BLOCKS)
        if bidx is None:
            return

        for doctor in self.doctors:
            panel = doctor.panel_patients
            if not panel:
                continue

            # Spread visits over the full 1-hour block (12 ticks)
            ticks_in_block = (DOCTOR_BLOCKS[bidx][1] - DOCTOR_BLOCKS[bidx][0]) // self.config.dt_min
            quota = math.ceil(len(panel) / ticks_in_block)
            start_offset = (tick - (DOCTOR_BLOCKS[bidx][0] // self.config.dt_min)) * quota
            end_offset = min(len(panel), start_offset + quota)

            for pid in panel[start_offset:end_offset]:
                pair = (doctor.unique_id, pid)
                if pair in self._doctor_block_seen_pairs[bidx]:
                    continue
                self._doctor_block_seen_pairs[bidx].add(pair)

                room_id = self.agent_index[pid].room_id
                self._record_event(
                    tick=tick,
                    actor_id=doctor.unique_id,
                    target_id=pid,
                    room_id=room_id,
                    event_type="doctor_round",
                )

    def _generate_nurse_rounds(self, tick: int, time_min: int):
        if not in_any_block(time_min, NURSE_ROUNDS_BLOCKS):
            return

        bidx = block_index(time_min, NURSE_ROUNDS_BLOCKS)
        block_start_tick = NURSE_ROUNDS_BLOCKS[bidx][0] // self.config.dt_min
        block_tick_offset = tick - block_start_tick

        for nurse in self.nurses:
            caseload_patients = []
            for rid in nurse.caseload_rooms:
                caseload_patients.extend(self.room_occupants[rid])

            if not caseload_patients:
                continue

            ticks_in_block = (NURSE_ROUNDS_BLOCKS[bidx][1] - NURSE_ROUNDS_BLOCKS[bidx][0]) // self.config.dt_min
            quota = max(1, math.ceil(len(caseload_patients) / ticks_in_block))
            start_idx = block_tick_offset * quota
            end_idx = min(len(caseload_patients), start_idx + quota)

            for pid in caseload_patients[start_idx:end_idx]:
                self._record_event(
                    tick=tick,
                    actor_id=nurse.unique_id,
                    target_id=pid,
                    room_id=self.agent_index[pid].room_id,
                    event_type="nurse_round",
                )

    def _generate_feeding_events(self, tick: int, time_min: int):
        bidx = block_index(time_min, FEEDING_BLOCKS)
        if bidx is None:
            return

        active_feeders = [n for n in self.nurses if n.is_active_feeder]
        if len(active_feeders) != 2:
            return

        # On first tick of each feeding block, sample coverage and split between feeders
        block_start_tick = FEEDING_BLOCKS[bidx][0] // self.config.dt_min
        if tick != block_start_tick:
            return

        coverage = self.rng.uniform(self.config.feeding_coverage_min, self.config.feeding_coverage_max)
        n_target = max(1, int(round(self.config.n_patients * coverage)))

        patient_ids = [p.unique_id for p in self.patients]
        selected = self.rng.sample(patient_ids, k=n_target)
        mid = len(selected) // 2
        assignments = {
            active_feeders[0].unique_id: selected[:mid],
            active_feeders[1].unique_id: selected[mid:],
        }

        # Emit events gradually over feeding block ticks
        ticks_in_block = (FEEDING_BLOCKS[bidx][1] - FEEDING_BLOCKS[bidx][0]) // self.config.dt_min
        self._feeding_schedule = getattr(self, "_feeding_schedule", {})
        self._feeding_schedule[bidx] = {
            "ticks_in_block": ticks_in_block,
            "start_tick": block_start_tick,
            "assignments": assignments,
        }

        # Also emit start tick chunk immediately
        self._emit_feeding_events_for_tick(tick, bidx)

    def _emit_feeding_events_for_tick(self, tick: int, bidx: int):
        if not hasattr(self, "_feeding_schedule") or bidx not in self._feeding_schedule:
            return

        sched = self._feeding_schedule[bidx]
        offset = tick - sched["start_tick"]
        if offset < 0 or offset >= sched["ticks_in_block"]:
            return

        for nurse_id, targets in sched["assignments"].items():
            if not targets:
                continue
            quota = max(1, math.ceil(len(targets) / sched["ticks_in_block"]))
            start_idx = offset * quota
            end_idx = min(len(targets), start_idx + quota)

            for pid in targets[start_idx:end_idx]:
                self._record_event(
                    tick=tick,
                    actor_id=nurse_id,
                    target_id=pid,
                    room_id=self.agent_index[pid].room_id,
                    event_type="feeding",
                )

    def _generate_ad_hoc(self, tick: int, time_min: int):
        if not in_block(time_min, AD_HOC_BLOCK):
            return

        if self.rng.random() > self.config.p_ad_hoc_tick:
            return

        n_events = self.rng.randint(0, self.config.ad_hoc_max_events_per_tick)
        if n_events == 0:
            return

        patient_ids = [p.unique_id for p in self.patients]
        staff_ids = [n.unique_id for n in self.nurses] + [d.unique_id for d in self.doctors]

        for _ in range(n_events):
            actor_id = self.rng.choice(staff_ids)
            target_id = self.rng.choice(patient_ids)
            self._record_event(
                tick=tick,
                actor_id=actor_id,
                target_id=target_id,
                room_id=self.agent_index[target_id].room_id,
                event_type="ad_hoc",
            )

    def _generate_roommate_events(self, tick: int, time_min: int):
        # Evaluate once per hour (every 12 ticks): at xx:00
        if time_min % 60 != 0:
            return

        hour_idx = time_min // 60
        for room_id, occupants in self.room_occupants.items():
            if len(occupants) < 2:
                continue

            key = (room_id, hour_idx)
            if key in self._room_hour_triggered:
                continue

            if self.rng.random() < self.config.p_roommate_event_per_room_per_hour:
                p1, p2 = self.rng.sample(occupants, k=2)
                self._record_event(
                    tick=tick,
                    actor_id=p1,
                    target_id=p2,
                    room_id=room_id,
                    event_type="roommate",
                )
                self._room_hour_triggered.add(key)

    def _generate_nurse_station_events(self, tick: int, time_min: int):
        is_handover = in_any_block(time_min, HANDOVER_BLOCKS)
        is_random_daytime_tick = tick in self._random_nurse_station_ticks

        if not (is_handover or is_random_daytime_tick):
            return

        # nurse-nurse
        nn_pairs = self.rng.randint(1, 3) if is_handover else self.rng.randint(0, 1)
        for _ in range(nn_pairs):
            n1, n2 = self.rng.sample(self.nurses, k=2)
            self._record_event(
                tick=tick,
                actor_id=n1.unique_id,
                target_id=n2.unique_id,
                room_id="nurse_station",
                event_type="nurse_station",
            )

        # nurse-doctor
        nd_pairs = self.rng.randint(1, 2) if is_handover else self.rng.randint(0, 1)
        for _ in range(nd_pairs):
            nurse = self.rng.choice(self.nurses)
            doctor = self.rng.choice(self.doctors)
            self._record_event(
                tick=tick,
                actor_id=nurse.unique_id,
                target_id=doctor.unique_id,
                room_id="nurse_station",
                event_type="nurse_station",
            )

        # doctor-doctor (rare)
        dd_pairs = 1 if (is_handover and self.rng.random() < 0.25) else 0
        for _ in range(dd_pairs):
            d1, d2 = self.rng.sample(self.doctors, k=2)
            self._record_event(
                tick=tick,
                actor_id=d1.unique_id,
                target_id=d2.unique_id,
                room_id="nurse_station",
                event_type="nurse_station",
            )

    def step(self):
        tick = self.current_tick
        time_min = tick_to_time_min(tick, self.config.dt_min)
        self.current_time_min = time_min

        self._generate_doctor_rounds(tick, time_min)
        self._generate_nurse_rounds(tick, time_min)

        # Feeding schedule generation + emission
        self._generate_feeding_events(tick, time_min)
        bidx = block_index(time_min, FEEDING_BLOCKS)
        if bidx is not None:
            self._emit_feeding_events_for_tick(tick, bidx)

        self._generate_ad_hoc(tick, time_min)
        self._generate_roommate_events(tick, time_min)
        self._generate_nurse_station_events(tick, time_min)

        self.current_tick += 1


# =========================================================
# 7) Run simulation (1 day, 288 ticks)
# =========================================================

def run_simulation(config: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = HospitalContactModel(config)

    for _ in range(config.ticks_per_day):
        model.step()

    visit_df = pd.DataFrame(model.visit_events)

    if visit_df.empty:
        agg_df = pd.DataFrame(
            columns=[
                "run_id",
                "u_id",
                "u_type",
                "v_id",
                "v_type",
                "total_contact_count",
                "first_time_min",
                "last_time_min",
            ]
        )
    else:
        agg_df = build_aggregated_edges(visit_df)

    summary_df = build_run_summary(config, visit_df, agg_df)

    return visit_df, agg_df, summary_df


# =========================================================
# 8) CSV export (visit_log, aggregated_edges, run_summary)
# =========================================================

def build_aggregated_edges(visit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in visit_df.iterrows():
        a_id, a_type = r["actor_id"], r["actor_type"]
        b_id, b_type = r["target_id"], r["target_type"]

        # undirected canonical ordering by id
        if a_id <= b_id:
            u_id, u_type, v_id, v_type = a_id, a_type, b_id, b_type
        else:
            u_id, u_type, v_id, v_type = b_id, b_type, a_id, a_type

        rows.append(
            {
                "run_id": r["run_id"],
                "u_id": u_id,
                "u_type": u_type,
                "v_id": v_id,
                "v_type": v_type,
                "time_min": int(r["time_min"]),
            }
        )

    tmp = pd.DataFrame(rows)
    agg = (
        tmp.groupby(["run_id", "u_id", "u_type", "v_id", "v_type"], as_index=False)
        .agg(
            total_contact_count=("time_min", "count"),
            first_time_min=("time_min", "min"),
            last_time_min=("time_min", "max"),
        )
        .sort_values(["total_contact_count", "u_id", "v_id"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return agg


def edge_type(actor_type: str, target_type: str) -> str:
    pair = sorted([actor_type[0].upper(), target_type[0].upper()])
    return "".join(pair)  # e.g. ['N','P'] -> 'NP'


def build_run_summary(config: SimConfig, visit_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    total_events = int(len(visit_df))
    unique_edges = int(len(agg_df))

    type_counts = Counter({"PP": 0, "NP": 0, "DP": 0, "NN": 0, "DN": 0, "DD": 0})

    if not visit_df.empty:
        for _, r in visit_df.iterrows():
            e = edge_type(r["actor_type"], r["target_type"])
            if e == "NP":
                type_counts["NP"] += 1
            elif e == "DP":
                type_counts["DP"] += 1
            elif e == "PP":
                type_counts["PP"] += 1
            elif e == "NN":
                type_counts["NN"] += 1
            elif e == "DN":
                type_counts["DN"] += 1
            elif e == "DD":
                type_counts["DD"] += 1

    # Degree / weighted degree from aggregated graph
    G = nx.Graph()
    if not agg_df.empty:
        for _, r in agg_df.iterrows():
            G.add_node(r["u_id"], role=r["u_type"])
            G.add_node(r["v_id"], role=r["v_type"])
            G.add_edge(r["u_id"], r["v_id"], weight=int(r["total_contact_count"]))

    degree = dict(G.degree())
    wdegree = dict(G.degree(weight="weight"))

    top5_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_wdegree = sorted(wdegree.items(), key=lambda x: x[1], reverse=True)[:5]

    summary = pd.DataFrame(
        [
            {
                "run_id": config.run_id,
                "seed": config.seed,
                "N_patients": config.n_patients,
                "N_nurses": config.n_nurses,
                "N_doctors": config.n_doctors,
                "N_rooms": config.n_rooms,
                "dt_min": config.dt_min,
                "ticks_per_day": config.ticks_per_day,
                "total_events": total_events,
                "unique_edges": unique_edges,
                "total_PP_events": int(type_counts["PP"]),
                "total_PN_events": int(type_counts["NP"]),
                "total_PD_events": int(type_counts["DP"]),
                "total_NN_events": int(type_counts["NN"]),
                "total_ND_events": int(type_counts["DN"]),
                "total_DD_events": int(type_counts["DD"]),
                "top5_nodes_by_degree": json.dumps(top5_degree),
                "top5_nodes_by_weighted_degree": json.dumps(top5_wdegree),
            }
        ]
    )
    return summary


def export_csvs(config: SimConfig, visit_df: pd.DataFrame, agg_df: pd.DataFrame, summary_df: pd.DataFrame):
    os.makedirs(config.output_dir, exist_ok=True)

    visit_path = os.path.join(config.output_dir, "visit_log.csv")
    agg_path = os.path.join(config.output_dir, "aggregated_edges.csv")
    summary_path = os.path.join(config.output_dir, "run_summary.csv")

    visit_df.to_csv(visit_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return visit_path, agg_path, summary_path


# =========================================================
# 9) Visualization (save 3 PNGs)
# =========================================================

def plot_network(config: SimConfig, agg_df: pd.DataFrame, out_path: str):
    G = nx.Graph()
    for _, r in agg_df.iterrows():
        G.add_node(r["u_id"], role=r["u_type"])
        G.add_node(r["v_id"], role=r["v_type"])
        G.add_edge(r["u_id"], r["v_id"], weight=int(r["total_contact_count"]))

    plt.figure(figsize=(14, 10))
    if G.number_of_nodes() == 0:
        plt.title("Contact Network (empty)")
        plt.axis("off")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        return

    pos = nx.spring_layout(G, seed=config.seed, k=0.45)

    role_to_color = {"patient": "#4C78A8", "nurse": "#F58518", "doctor": "#54A24B"}
    node_colors = [role_to_color.get(G.nodes[n].get("role", "patient"), "gray") for n in G.nodes()]

    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    edge_widths = [min(0.5 + w * 0.15, 6.0) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=220, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35)

    # Reduce clutter by labeling only staff nodes
    staff_nodes = [n for n, d in G.nodes(data=True) if d.get("role") in {"nurse", "doctor"}]
    labels = {n: n for n in staff_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title("Aggregated Contact Network (Undirected)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_timeseries(config: SimConfig, visit_df: pd.DataFrame, out_path: str):
    ticks = list(range(config.ticks_per_day))
    total_counts = [0] * config.ticks_per_day
    pn_counts = [0] * config.ticks_per_day
    pd_counts = [0] * config.ticks_per_day

    if not visit_df.empty:
        for _, r in visit_df.iterrows():
            tick = int(r["tick"])
            total_counts[tick] += 1
            tpair = edge_type(r["actor_type"], r["target_type"])
            if tpair == "NP":
                pn_counts[tick] += 1
            elif tpair == "DP":
                pd_counts[tick] += 1

    x_hours = [t * config.dt_min / 60.0 for t in ticks]

    plt.figure(figsize=(14, 5))
    plt.plot(x_hours, total_counts, label="Total events", linewidth=1.8)
    plt.plot(x_hours, pn_counts, label="PN events", linewidth=1.4)
    plt.plot(x_hours, pd_counts, label="PD events", linewidth=1.4)
    plt.xlabel("Hour of day")
    plt.ylabel("Events per 5-min tick")
    plt.title("Contact Events Time Series")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_degree_hist(config: SimConfig, agg_df: pd.DataFrame, out_path: str):
    G = nx.Graph()
    for _, r in agg_df.iterrows():
        G.add_node(r["u_id"], role=r["u_type"])
        G.add_node(r["v_id"], role=r["v_type"])
        G.add_edge(r["u_id"], r["v_id"], weight=int(r["total_contact_count"]))

    roles = {"patient": [], "nurse": [], "doctor": []}
    wroles = {"patient": [], "nurse": [], "doctor": []}

    for node, role in nx.get_node_attributes(G, "role").items():
        roles[role].append(G.degree(node))
        wroles[role].append(G.degree(node, weight="weight"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"patient": "#4C78A8", "nurse": "#F58518", "doctor": "#54A24B"}

    for role in ["patient", "nurse", "doctor"]:
        if roles[role]:
            axes[0].hist(roles[role], alpha=0.6, bins=15, label=role, color=colors[role])
        if wroles[role]:
            axes[1].hist(wroles[role], alpha=0.6, bins=15, label=role, color=colors[role])

    axes[0].set_title("Degree distribution by role")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].set_title("Weighted degree distribution by role")
    axes[1].set_xlabel("Weighted degree (contact count)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def export_figures(config: SimConfig, visit_df: pd.DataFrame, agg_df: pd.DataFrame):
    fig_dir = os.path.join(config.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    network_path = os.path.join(fig_dir, "network.png")
    timeseries_path = os.path.join(fig_dir, "timeseries.png")
    degree_hist_path = os.path.join(fig_dir, "degree_hist.png")

    plot_network(config, agg_df, network_path)
    plot_timeseries(config, visit_df, timeseries_path)
    plot_degree_hist(config, agg_df, degree_hist_path)

    return network_path, timeseries_path, degree_hist_path


# =========================================================
# 10) CLI (seed/run_id) + console summary
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mesa hospital contact-network prototype (no infection)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Run identifier (default: UNIX timestamp)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_id = args.run_id or str(int(datetime.utcnow().timestamp()))
    config = SimConfig(seed=args.seed, run_id=run_id)

    visit_df, agg_df, summary_df = run_simulation(config)
    visit_path, agg_path, summary_path = export_csvs(config, visit_df, agg_df, summary_df)
    net_path, ts_path, deg_path = export_figures(config, visit_df, agg_df)

    total_events = int(summary_df.loc[0, "total_events"])
    unique_edges = int(summary_df.loc[0, "unique_edges"])

    print("\n=== Simulation finished ===")
    print(f"run_id={config.run_id} | seed={config.seed}")
    print(f"total_events={total_events}, unique_edges={unique_edges}")
    print("\nOutput files:")
    print(f"- {visit_path}")
    print(f"- {agg_path}")
    print(f"- {summary_path}")
    print(f"- {net_path}")
    print(f"- {ts_path}")
    print(f"- {deg_path}")


if __name__ == "__main__":
    main()
