#!/usr/bin/env python3
"""
Single-file Mesa prototype: 1 ward hospital contact network generation (NO infection dynamics).

REFACTORED VERSION: More agent-centric behavior with explicit agent-level state machines.
Nurses and doctors now have their own step() methods and decide their behavior based on
time blocks and assignments. The model coordinates the environment and event logging,
but agents drive most contact generation.

MULTI-DAY EXTENSION:
- 30-day simulation
- dynamic admissions/discharges
- fixed baseline staffing
- no infection dynamics yet

Outputs:
- outputs/visit_log.csv
- outputs/aggregated_edges.csv
- outputs/run_summary.csv
- outputs/figures/network.png
- outputs/figures/timeseries.png
- outputs/figures/degree_hist.png
"""

# =========================================================
# 1) IMPORTS & CONSTANTS
# =========================================================

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mesa import Agent, Model


DEFAULT_SEED = 42
DEFAULT_DT_MIN = 5
MINUTES_PER_DAY = 24 * 60
TICKS_PER_DAY = MINUTES_PER_DAY // DEFAULT_DT_MIN

# 83-bed ward:
# 1x1-bed, 5x2-bed, 18x4-bed = 83 beds total
ROOM_CAPACITY_SPEC = [1, 2, 2, 2, 2, 2] + [4] * 18
N_PATIENTS = sum(ROOM_CAPACITY_SPEC)  # 83
N_ROOMS = len(ROOM_CAPACITY_SPEC)     # 24

# Fixed baseline staffing
N_NURSES = 7
N_DOCTORS = 4

DURATION_MIN_DEFAULT = 5

# Scheduled care activity time blocks [start_minute, end_minute)
DOCTOR_BLOCKS = [(9 * 60, 10 * 60)]  # one doctor round per day
NURSE_ROUNDS_BLOCKS = [(6 * 60, 7 * 60), (12 * 60, 13 * 60), (16 * 60, 17 * 60)]  # 3 rounds/day
FEEDING_BLOCKS = [(8 * 60, 9 * 60), (12 * 60, 13 * 60), (18 * 60, 19 * 60)]
AD_HOC_BLOCK = (9 * 60, 15 * 60)
HANDOVER_BLOCKS = [(6 * 60 + 55, 7 * 60 + 5), (18 * 60 + 55, 19 * 60 + 5)]


# =========================================================
# 2) PARAMETERS & CONFIGURATION
# =========================================================
@dataclass
@dataclass
class SimConfig:
    seed: int = DEFAULT_SEED
    run_id: str = ""

    dt_min: int = DEFAULT_DT_MIN
    ticks_per_day: int = TICKS_PER_DAY
    simulation_days: int = 30

    ward_capacity: int = N_PATIENTS
    n_patients: int = N_PATIENTS
    n_nurses: int = N_NURSES
    n_doctors: int = N_DOCTORS
    n_rooms: int = N_ROOMS

    target_bed_occupancy: float = 0.6829
    mean_los_days: float = 7.69
    initial_patient_count: int = 57
    daily_admissions_mean: float = 7.37
    los_distribution: str = "fixed"
    initial_remaining_los_distribution: str = "discrete_uniform_1_8"

    baseline_nurses_day: int = 7
    baseline_nurses_night: int = 5
    baseline_doctors_day: int = 4
    baseline_doctors_night: int = 1

    shift_length_hours: int = 12
    nurse_shift_times: str = "07:00-19:00"
    doctor_shift_times: str = "08:00-17:00"

    nurse_rounds_per_day: int = 3
    doctor_visits_per_patient_per_day: int = 1

    day_activity_share: float = 0.941
    night_activity_share: float = 0.059

    mean_nurse_patient_contact_duration_min: int = 10
    mean_doctor_patient_contact_duration_min: int = 4
    contact_duration_distribution: str = "exponential"

    admission_room_assignment_rule: str = "first_available_bed"
    day_boundary_rule: str = "decrement_los_then_discharge_then_admit_then_refresh_assignments"

    feeding_coverage_min: float = 0.30
    feeding_coverage_max: float = 0.50
    p_ad_hoc_tick: float = 0.20
    ad_hoc_max_events_per_tick: int = 2
    p_roommate_event_per_room_per_hour: float = 0.10
    nurse_station_random_ticks_per_day: int = 10

    output_dir: str = "outputs"


# =========================================================
# 3) TIME UTILITIES
# =========================================================
def build_run_output_dir(base_output_dir: str, run_id: str, seed: int) -> str:
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp_str}_seed{seed}_{run_id}"
    return os.path.join(base_output_dir, run_dir_name)


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
# 4) AGENT CLASSES
# =========================================================
class BaseHospitalAgent(Agent):
    def __init__(self, model: Model, unique_id: str, agent_type: str):
        super().__init__(model)
        self.unique_id = unique_id
        self.agent_type = agent_type


class PatientAgent(BaseHospitalAgent):
    """
    Passive patient agent with multi-day stay attributes.
    """
    def __init__(
        self,
        model: Model,
        unique_id: str,
        room_id: str,
        admission_day: int | None = None,
        remaining_los_days: int | None = None,
        is_active: bool = False,
    ):
        super().__init__(model, unique_id, "patient")
        self.room_id = room_id
        self.admission_day = admission_day
        self.remaining_los_days = remaining_los_days
        self.is_active = is_active

    def step(self):
        if not self.is_active:
            return
        pass


class NurseAgent(BaseHospitalAgent):
    def __init__(self, model: Model, unique_id: str):
        super().__init__(model, unique_id, "nurse")
        self.caseload_rooms: list[str] = []
        self.is_active_feeder: bool = False

        self.current_state: str = "idle"

        self.round_visits_per_tick: dict[int, list[str]] = {}
        self.round_block_started_at_tick: int = -1

        self.feeding_visits_per_tick: dict[int, list[str]] = {}
        self.feeding_block_started_at_tick: int = -1

        self.handover_block_idx: int = -1
        self.handover_initiated_this_block: bool = False

    def step(self):
        time_min = self.model.get_current_time_min()
        tick = self.model.current_tick

        self.current_state = self._get_current_state(time_min)

        def _is_active_patient(pid: str) -> bool:
            p = self.model.agent_index.get(pid)
            return (p is not None) and getattr(p, "is_active", False) is True

        if self.current_state == "rounding":
            if tick in self.round_visits_per_tick:
                self.round_visits_per_tick[tick] = [
                    pid for pid in self.round_visits_per_tick[tick] if _is_active_patient(pid)
                ]
            self._handle_rounding(tick, time_min)

        elif self.current_state == "feeding":
            if tick in self.feeding_visits_per_tick:
                self.feeding_visits_per_tick[tick] = [
                    pid for pid in self.feeding_visits_per_tick[tick] if _is_active_patient(pid)
                ]
            self._handle_feeding(tick, time_min)

        elif self.current_state == "handover":
            self._handle_handover(tick, time_min)

        elif self.current_state == "ad_hoc":
            has_active = False
            for rid in self.caseload_rooms:
                for pid in self.model.get_patients_in_room(rid):
                    if _is_active_patient(pid):
                        has_active = True
                        break
                if has_active:
                    break
            if has_active:
                self._handle_ad_hoc(tick, time_min)

    def _get_current_state(self, time_min: int) -> str:
        if in_any_block(time_min, HANDOVER_BLOCKS):
            return "handover"
        if self.is_active_feeder and in_any_block(time_min, FEEDING_BLOCKS):
            return "feeding"
        if in_any_block(time_min, NURSE_ROUNDS_BLOCKS):
            return "rounding"
        if in_block(time_min, AD_HOC_BLOCK):
            return "ad_hoc"
        return "station"

    def prepare_round_block(self, tick: int, time_min: int):
        bidx = block_index(time_min, NURSE_ROUNDS_BLOCKS)
        if bidx is None:
            return
        if self.round_block_started_at_tick == tick:
            return

        caseload_patients = []
        for rid in self.caseload_rooms:
            caseload_patients.extend(self.model.get_patients_in_room(rid))
        self.model.rng.shuffle(caseload_patients)

        block_start_tick = NURSE_ROUNDS_BLOCKS[bidx][0] // self.model.config.dt_min
        block_end_tick = NURSE_ROUNDS_BLOCKS[bidx][1] // self.model.config.dt_min

        self.round_visits_per_tick = {}
        remaining_patients = list(caseload_patients)

        for current_tick in range(block_start_tick, block_end_tick):
            ticks_left = block_end_tick - current_tick
            patients_left = len(remaining_patients)

            if patients_left == 0:
                break

            n_to_visit = max(1, math.ceil(patients_left / ticks_left))
            jitter = self.model.rng.randint(-1, 2)
            n_to_visit = max(1, min(n_to_visit + jitter, patients_left))

            visits_this_tick = self.model.rng.sample(remaining_patients, k=n_to_visit)
            self.round_visits_per_tick[current_tick] = visits_this_tick

            for pid in visits_this_tick:
                remaining_patients.remove(pid)

        self.round_block_started_at_tick = tick

    def prepare_feeding_block(self, tick: int, time_min: int):
        if not self.is_active_feeder:
            return

        bidx = block_index(time_min, FEEDING_BLOCKS)
        if bidx is None:
            return
        if self.feeding_block_started_at_tick == tick:
            return

        assigned = self.model.get_feeding_assignment_for_nurse(self.unique_id, bidx)
        self.model.rng.shuffle(assigned)

        block_start_tick = FEEDING_BLOCKS[bidx][0] // self.model.config.dt_min
        block_end_tick = FEEDING_BLOCKS[bidx][1] // self.model.config.dt_min

        self.feeding_visits_per_tick = {}
        remaining_patients = list(assigned)

        for current_tick in range(block_start_tick, block_end_tick):
            ticks_left = block_end_tick - current_tick
            patients_left = len(remaining_patients)

            if patients_left == 0:
                break

            n_to_feed = max(1, math.ceil(patients_left / ticks_left))
            jitter = self.model.rng.randint(-1, 2)
            n_to_feed = max(1, min(n_to_feed + jitter, patients_left))

            feedings_this_tick = self.model.rng.sample(remaining_patients, k=n_to_feed)
            self.feeding_visits_per_tick[current_tick] = feedings_this_tick

            for pid in feedings_this_tick:
                remaining_patients.remove(pid)

        self.feeding_block_started_at_tick = tick

    def _handle_rounding(self, tick: int, time_min: int):
        bidx = block_index(time_min, NURSE_ROUNDS_BLOCKS)
        if bidx is None:
            return

        block_start_tick = NURSE_ROUNDS_BLOCKS[bidx][0] // self.model.config.dt_min
        if tick == block_start_tick:
            self.prepare_round_block(tick, time_min)

        if tick in self.round_visits_per_tick:
            for pid in self.round_visits_per_tick[tick]:
                if self.model.is_patient_active(pid):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=pid,
                        event_type="nurse_round",
                    )

    def _handle_feeding(self, tick: int, time_min: int):
        if not self.is_active_feeder:
            return

        bidx = block_index(time_min, FEEDING_BLOCKS)
        if bidx is None:
            return

        block_start_tick = FEEDING_BLOCKS[bidx][0] // self.model.config.dt_min
        if tick == block_start_tick:
            self.prepare_feeding_block(tick, time_min)

        if tick in self.feeding_visits_per_tick:
            for pid in self.feeding_visits_per_tick[tick]:
                if self.model.is_patient_active(pid):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=pid,
                        event_type="feeding",
                    )

    def _handle_handover(self, tick: int, time_min: int):
        bidx = block_index(time_min, HANDOVER_BLOCKS)
        if bidx is None:
            if self.handover_block_idx != -1:
                self.handover_block_idx = -1
                self.handover_initiated_this_block = False
            return

        if bidx != self.handover_block_idx:
            self.handover_block_idx = bidx
            self.handover_initiated_this_block = False

        if self.handover_initiated_this_block:
            return
        if self.model.rng.random() > 0.6:
            return

        interactant_option = self.model.rng.choice(["nurse", "doctor"])

        if interactant_option == "nurse":
            other_nurses = [n for n in self.model.nurses if n.unique_id != self.unique_id]
            if other_nurses:
                other = self.model.rng.choice(other_nurses)
                if not self.model.is_recent_contact(self.unique_id, other.unique_id, tick, window_ticks=5):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=other.unique_id,
                        event_type="nurse_station",
                    )
                    self.handover_initiated_this_block = True
        else:
            if self.model.doctors:
                doctor = self.model.rng.choice(self.model.doctors)
                if not self.model.is_recent_contact(self.unique_id, doctor.unique_id, tick, window_ticks=5):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=doctor.unique_id,
                        event_type="nurse_station",
                    )
                    self.handover_initiated_this_block = True

    def _handle_ad_hoc(self, tick: int, time_min: int):
        if not in_block(time_min, AD_HOC_BLOCK):
            return
        if self.model.rng.random() > (self.model.config.p_ad_hoc_tick * 0.3):
            return

        if self.caseload_rooms:
            caseload_patients = []
            for rid in self.caseload_rooms:
                caseload_patients.extend(self.model.get_patients_in_room(rid))
            caseload_patients = [pid for pid in caseload_patients if self.model.is_patient_active(pid)]

            if caseload_patients:
                pid = self.model.rng.choice(caseload_patients)
                if not self.model.is_recent_contact(self.unique_id, pid, tick, window_ticks=24):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=pid,
                        event_type="ad_hoc",
                    )


class DoctorAgent(BaseHospitalAgent):
    def __init__(self, model: Model, unique_id: str):
        super().__init__(model, unique_id, "doctor")
        self.panel_patients: list[str] = []

        self.current_state: str = "idle"

        self.round_visits_per_tick: dict[int, list[str]] = {}
        self.round_block_started_at_tick: int = -1

        self.handover_block_idx: int = -1
        self.handover_initiated_this_block: bool = False

    def step(self):
        time_min = self.model.get_current_time_min()
        tick = self.model.current_tick

        self.current_state = self._get_current_state(time_min)

        if self.current_state == "rounding":
            self._handle_rounding(tick, time_min)
        elif self.current_state == "handover":
            self._handle_handover(tick, time_min)
        elif self.current_state == "ad_hoc":
            self._handle_ad_hoc(tick, time_min)

    def _get_current_state(self, time_min: int) -> str:
        if in_any_block(time_min, DOCTOR_BLOCKS):
            return "rounding"
        if in_any_block(time_min, HANDOVER_BLOCKS):
            return "handover"
        if in_block(time_min, AD_HOC_BLOCK):
            return "ad_hoc"
        return "station"

    def prepare_round_block(self, tick: int, time_min: int):
        bidx = block_index(time_min, DOCTOR_BLOCKS)
        if bidx is None:
            return
        if self.round_block_started_at_tick == tick:
            return

        panel_copy = [pid for pid in self.panel_patients if self.model.is_patient_active(pid)]
        self.model.rng.shuffle(panel_copy)

        block_start_tick = DOCTOR_BLOCKS[bidx][0] // self.model.config.dt_min
        block_end_tick = DOCTOR_BLOCKS[bidx][1] // self.model.config.dt_min

        self.round_visits_per_tick = {}
        remaining_patients = list(panel_copy)

        for current_tick in range(block_start_tick, block_end_tick):
            ticks_left = block_end_tick - current_tick
            patients_left = len(remaining_patients)

            if patients_left == 0:
                break

            n_to_visit = max(1, math.ceil(patients_left / ticks_left))
            jitter = self.model.rng.randint(-1, 2)
            n_to_visit = max(1, min(n_to_visit + jitter, patients_left))

            visits_this_tick = self.model.rng.sample(remaining_patients, k=n_to_visit)
            self.round_visits_per_tick[current_tick] = visits_this_tick

            for pid in visits_this_tick:
                remaining_patients.remove(pid)

        self.round_block_started_at_tick = tick

    def _handle_rounding(self, tick: int, time_min: int):
        bidx = block_index(time_min, DOCTOR_BLOCKS)
        if bidx is None:
            return

        block_start_tick = DOCTOR_BLOCKS[bidx][0] // self.model.config.dt_min
        if tick == block_start_tick:
            self.prepare_round_block(tick, time_min)

        if tick in self.round_visits_per_tick:
            for pid in self.round_visits_per_tick[tick]:
                if self.model.is_patient_active(pid):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=pid,
                        event_type="doctor_round",
                    )

    def _handle_handover(self, tick: int, time_min: int):
        bidx = block_index(time_min, HANDOVER_BLOCKS)
        if bidx is None:
            if self.handover_block_idx != -1:
                self.handover_block_idx = -1
                self.handover_initiated_this_block = False
            return

        if bidx != self.handover_block_idx:
            self.handover_block_idx = bidx
            self.handover_initiated_this_block = False

        if self.handover_initiated_this_block:
            return
        if self.model.rng.random() > 0.5:
            return

        interactant_option = self.model.rng.choice(["doctor", "nurse"])

        if interactant_option == "doctor":
            other_doctors = [d for d in self.model.doctors if d.unique_id != self.unique_id]
            if other_doctors:
                other = self.model.rng.choice(other_doctors)
                if not self.model.is_recent_contact(self.unique_id, other.unique_id, tick, window_ticks=5):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=other.unique_id,
                        event_type="nurse_station",
                    )
                    self.handover_initiated_this_block = True
        else:
            if self.model.nurses:
                nurse = self.model.rng.choice(self.model.nurses)
                if not self.model.is_recent_contact(self.unique_id, nurse.unique_id, tick, window_ticks=5):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=nurse.unique_id,
                        event_type="nurse_station",
                    )
                    self.handover_initiated_this_block = True

    def _handle_ad_hoc(self, tick: int, time_min: int):
        if not in_block(time_min, AD_HOC_BLOCK):
            return
        if self.model.rng.random() > (self.model.config.p_ad_hoc_tick * 0.15):
            return
        if not self.panel_patients:
            return

        active_panel = [pid for pid in self.panel_patients if self.model.is_patient_active(pid)]
        if not active_panel:
            return

        pid = self.model.rng.choice(active_panel)
        if not self.model.is_recent_contact(self.unique_id, pid, tick, window_ticks=24):
            self.model.record_contact(
                tick=tick,
                actor_id=self.unique_id,
                target_id=pid,
                event_type="ad_hoc",
            )


# =========================================================
# 5) MESA MODEL
# =========================================================
class HospitalContactModel(Model):
    def __init__(self, config: SimConfig):
        super().__init__()
        self.config = config
        self.rng = random.Random(config.seed)

        self.room_capacity_map: dict[str, int] = self._build_room_capacity_map()
        self.room_occupants: dict[str, list[str]] = {rid: [] for rid in self.room_capacity_map}

        self.patients: list[PatientAgent] = []
        self.nurses: list[NurseAgent] = []
        self.doctors: list[DoctorAgent] = []
        self.agent_index: dict[str, BaseHospitalAgent] = {}

        self.visit_events: list[dict] = []

        self.current_tick = 0
        self.current_time_min = 0

        self._feeding_block_assignments: dict[int, dict[str, list[str]]] = {}
        self._room_hour_triggered: set[tuple[str, int]] = set()
        self._recent_contacts: dict[tuple[str, str], int] = {}
        self._random_nurse_station_ticks = self._sample_random_nurse_station_ticks()

        # Multi-day tracking
        self.total_admissions = 0
        self.total_discharges = 0
        self.daily_census_history: list[int] = []

        self._init_agents()
        self._assign_patients_to_rooms_deterministic()
        self._assign_nurse_room_caseloads()
        self._assign_doctor_panels()
        self._assign_daily_feeders()

        self.daily_census_history.append(self.get_current_patient_count())

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
        initial_n = self.config.initial_patient_count
        active_patients = self.patients[:initial_n]

        for p in active_patients:
            p.is_active = True
            p.admission_day = 0
            p.remaining_los_days = self.rng.randint(1, 8)

        patient_iter = iter(active_patients)
        for room_id, capacity in self.room_capacity_map.items():
            for _ in range(capacity):
                try:
                    patient = next(patient_iter)
                except StopIteration:
                    return
                patient.room_id = room_id
                self.room_occupants[room_id].append(patient.unique_id)

    def _assign_nurse_room_caseloads(self):
        for nurse in self.nurses:
            nurse.caseload_rooms = []

        room_ids = sorted(self.room_capacity_map.keys(), key=lambda x: int(x.split("_")[1]))
        occupied_rooms = [rid for rid in room_ids if len(self.room_occupants[rid]) > 0]

        for i, room_id in enumerate(occupied_rooms):
            nurse = self.nurses[i % len(self.nurses)]
            nurse.caseload_rooms.append(room_id)

    def _assign_doctor_panels(self):
        for doctor in self.doctors:
            doctor.panel_patients = []

        active_patient_ids = [p.unique_id for p in self.patients if p.is_active]
        for i, pid in enumerate(active_patient_ids):
            doctor = self.doctors[i % len(self.doctors)]
            doctor.panel_patients.append(pid)

    def _assign_daily_feeders(self):
        for nurse in self.nurses:
            nurse.is_active_feeder = False

        k = min(2, len(self.nurses))
        feeder_indices = self.rng.sample(range(len(self.nurses)), k=k)
        for i, nurse in enumerate(self.nurses):
            nurse.is_active_feeder = i in feeder_indices

    def _sample_random_nurse_station_ticks(self) -> set[int]:
        daytime_tick_start = AD_HOC_BLOCK[0] // self.config.dt_min
        daytime_tick_end = AD_HOC_BLOCK[1] // self.config.dt_min
        all_daytime_ticks = list(range(daytime_tick_start, daytime_tick_end))

        k = min(self.config.nurse_station_random_ticks_per_day, len(all_daytime_ticks))
        return set(self.rng.sample(all_daytime_ticks, k=k))

    # =========================================================
    # Helper methods
    # =========================================================
    def get_current_time_min(self) -> int:
        return self.current_time_min

    def get_current_day(self) -> int:
        return self.current_tick // self.config.ticks_per_day

    def get_patients_in_room(self, room_id: str) -> list[str]:
        return self.room_occupants.get(room_id, [])

    def get_current_patient_count(self) -> int:
        return sum(1 for p in self.patients if p.is_active)

    def is_patient_active(self, patient_id: str) -> bool:
        p = self.agent_index.get(patient_id)
        return isinstance(p, PatientAgent) and p.is_active

    def _sample_poisson(self, lam: float) -> int:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1

    def _find_first_available_bed(self) -> str | None:
        for room_id, capacity in self.room_capacity_map.items():
            if len(self.room_occupants[room_id]) < capacity:
                return room_id
        return None

    def _get_inactive_patients_pool(self) -> list[PatientAgent]:
        return [p for p in self.patients if not p.is_active]

    def _discharge_patient(self, patient: PatientAgent):
        if patient.room_id and patient.unique_id in self.room_occupants[patient.room_id]:
            self.room_occupants[patient.room_id].remove(patient.unique_id)

        patient.is_active = False
        patient.room_id = ""
        patient.admission_day = None
        patient.remaining_los_days = None
        self.total_discharges += 1

    def _admit_patient(self, patient: PatientAgent, room_id: str):
        patient.is_active = True
        patient.room_id = room_id
        patient.admission_day = self.get_current_day()
        patient.remaining_los_days = 8
        self.room_occupants[room_id].append(patient.unique_id)
        self.total_admissions += 1

    def _refresh_assignments_after_census_change(self):
        for nurse in self.nurses:
            nurse.caseload_rooms = []
            nurse.round_visits_per_tick = {}
            nurse.feeding_visits_per_tick = {}
            nurse.round_block_started_at_tick = -1
            nurse.feeding_block_started_at_tick = -1
            nurse.is_active_feeder = False

        for doctor in self.doctors:
            doctor.panel_patients = []
            doctor.round_visits_per_tick = {}
            doctor.round_block_started_at_tick = -1

        self._feeding_block_assignments = {}
        self._assign_nurse_room_caseloads()
        self._assign_doctor_panels()
        self._assign_daily_feeders()

    def _run_day_boundary_update(self):
        active_patients = [p for p in self.patients if p.is_active]

        for patient in active_patients:
            if patient.remaining_los_days is not None:
                patient.remaining_los_days -= 1

        for patient in list(active_patients):
            if patient.remaining_los_days is not None and patient.remaining_los_days <= 0:
                self._discharge_patient(patient)

        daily_admissions = self._sample_poisson(self.config.daily_admissions_mean)
        inactive_pool = self._get_inactive_patients_pool()

        for patient in inactive_pool:
            if daily_admissions <= 0:
                break
            room_id = self._find_first_available_bed()
            if room_id is None:
                break
            self._admit_patient(patient, room_id)
            daily_admissions -= 1

        self._refresh_assignments_after_census_change()
        self.daily_census_history.append(self.get_current_patient_count())

    def _sample_contact_duration(self, event_type: str, actor_type: str, target_type: str) -> int:
        if actor_type == "nurse" and target_type == "patient":
            mean = self.config.mean_nurse_patient_contact_duration_min
        elif actor_type == "doctor" and target_type == "patient":
            mean = self.config.mean_doctor_patient_contact_duration_min
        else:
            mean = DURATION_MIN_DEFAULT

        if self.config.contact_duration_distribution == "exponential":
            duration = max(1, int(round(self.rng.expovariate(1 / mean))))
            return duration
        return mean

    def record_contact(
        self,
        tick: int,
        actor_id: str,
        target_id: str,
        event_type: str,
        duration_min: int | None = None,
    ):
        actor = self.agent_index[actor_id]
        target = self.agent_index[target_id]
        time_min = tick_to_time_min(tick, self.config.dt_min)

        if target.agent_type == "patient":
            room_id = target.room_id
            if isinstance(target, PatientAgent) and not target.is_active:
                return
        else:
            room_id = "nurse_station"

        if duration_min is None:
            duration_min = self._sample_contact_duration(event_type, actor.agent_type, target.agent_type)

        event = {
            "run_id": self.config.run_id,
            "tick": tick,
            "day": self.get_current_day(),
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

        contact_pair = tuple(sorted([actor_id, target_id]))
        self._recent_contacts[contact_pair] = tick

    def is_tick_in_block(self, tick: int, block: tuple[int, int]) -> bool:
        time_min = tick_to_time_min(tick, self.config.dt_min)
        return in_block(time_min, block)

    def is_recent_contact(self, actor_id: str, target_id: str, current_tick: int, window_ticks: int = 12) -> bool:
        contact_pair = tuple(sorted([actor_id, target_id]))
        if contact_pair not in self._recent_contacts:
            return False

        last_contact_tick = self._recent_contacts[contact_pair]
        return (current_tick - last_contact_tick) < window_ticks

    def get_feeding_assignment_for_nurse(self, nurse_id: str, bidx: int) -> list[str]:
        if bidx not in self._feeding_block_assignments:
            coverage = self.rng.uniform(self.config.feeding_coverage_min, self.config.feeding_coverage_max)
            active_patient_ids = [p.unique_id for p in self.patients if p.is_active]
            n_target = max(1, int(round(len(active_patient_ids) * coverage))) if active_patient_ids else 0

            selected = self.rng.sample(active_patient_ids, k=min(n_target, len(active_patient_ids))) if active_patient_ids else []

            active_feeders = [n for n in self.nurses if n.is_active_feeder]
            if len(active_feeders) == 2:
                mid = len(selected) // 2
                self._feeding_block_assignments[bidx] = {
                    active_feeders[0].unique_id: selected[:mid],
                    active_feeders[1].unique_id: selected[mid:],
                }
            else:
                self._feeding_block_assignments[bidx] = {nurse_id: selected}

        return self._feeding_block_assignments[bidx].get(nurse_id, [])

    # =========================================================
    # Model-level event generation
    # =========================================================
    def _generate_roommate_events(self, tick: int, time_min: int):
        if time_min % 60 != 0:
            return

        hour_idx = time_min // 60
        for room_id, occupants in self.room_occupants.items():
            active_occupants = [pid for pid in occupants if self.is_patient_active(pid)]
            if len(active_occupants) < 2:
                continue

            key = (room_id, hour_idx)
            if key in self._room_hour_triggered:
                continue

            if self.rng.random() < self.config.p_roommate_event_per_room_per_hour:
                p1, p2 = self.rng.sample(active_occupants, k=2)
                event = {
                    "run_id": self.config.run_id,
                    "tick": tick,
                    "day": self.get_current_day(),
                    "time_min": tick_to_time_min(tick % self.config.ticks_per_day, self.config.dt_min),
                    "time_str": minutes_to_hhmm(tick_to_time_min(tick % self.config.ticks_per_day, self.config.dt_min)),
                    "actor_id": p1,
                    "actor_type": "patient",
                    "target_id": p2,
                    "target_type": "patient",
                    "room_id": room_id,
                    "event_type": "roommate",
                    "duration_min": DURATION_MIN_DEFAULT,
                }
                self.visit_events.append(event)
                self._room_hour_triggered.add(key)

    def _generate_nurse_station_events(self, tick: int, time_min: int):
        is_random_daytime_tick = (tick % self.config.ticks_per_day) in self._random_nurse_station_ticks
        if not is_random_daytime_tick:
            return

        if self.rng.random() < 0.3 and len(self.nurses) >= 2:
            n1, n2 = self.rng.sample(self.nurses, k=2)
            if not self.is_recent_contact(n1.unique_id, n2.unique_id, tick, window_ticks=6):
                self.record_contact(
                    tick=tick,
                    actor_id=n1.unique_id,
                    target_id=n2.unique_id,
                    event_type="nurse_station",
                    duration_min=DURATION_MIN_DEFAULT,
                )

    def step(self):
        tick = self.current_tick

        if tick > 0 and tick % self.config.ticks_per_day == 0:
            self._run_day_boundary_update()

        time_min = tick_to_time_min(tick % self.config.ticks_per_day, self.config.dt_min)
        self.current_time_min = time_min

        for patient in self.patients:
            if patient.is_active:
                patient.step()

        for nurse in self.nurses:
            nurse.step()

        for doctor in self.doctors:
            doctor.step()

        self._generate_roommate_events(tick, time_min)
        self._generate_nurse_station_events(tick, time_min)

        self.current_tick += 1


# =========================================================
# 6) SIMULATION EXECUTION
# =========================================================
def run_simulation(config: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = HospitalContactModel(config)

    total_ticks = config.ticks_per_day * config.simulation_days
    for _ in range(total_ticks):
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

    summary_df = build_run_summary(config, model, visit_df, agg_df)
    return visit_df, agg_df, summary_df


# =========================================================
# 7) DATA AGGREGATION & SUMMARY
# =========================================================
def build_aggregated_edges(visit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in visit_df.iterrows():
        a_id, a_type = r["actor_id"], r["actor_type"]
        b_id, b_type = r["target_id"], r["target_type"]

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
    return "".join(pair)


def build_run_summary(
    config: SimConfig,
    model: HospitalContactModel,
    visit_df: pd.DataFrame,
    agg_df: pd.DataFrame,
) -> pd.DataFrame:
    total_events = int(len(visit_df))
    unique_edges = int(len(agg_df))

    type_counts = Counter({"PP": 0, "NP": 0, "DP": 0, "NN": 0, "DN": 0, "DD": 0})

    if not visit_df.empty:
        for _, r in visit_df.iterrows():
            e = edge_type(r["actor_type"], r["target_type"])
            type_counts[e] += 1

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

    census_history = model.daily_census_history if model.daily_census_history else [model.get_current_patient_count()]
    occupancy_history = [c / config.n_patients for c in census_history]

    summary = pd.DataFrame(
        [
            {
                "run_id": config.run_id,
                "seed": config.seed,
                "N_patients_capacity": config.n_patients,
                "N_nurses": config.n_nurses,
                "N_doctors": config.n_doctors,
                "N_rooms": config.n_rooms,
                "dt_min": config.dt_min,
                "ticks_per_day": config.ticks_per_day,
                "simulation_days": config.simulation_days,
                "initial_patient_count": config.initial_patient_count,
                "total_admissions": model.total_admissions,
                "total_discharges": model.total_discharges,
                "final_patient_count": model.get_current_patient_count(),
                "average_daily_census": float(sum(census_history) / len(census_history)),
                "occupancy_mean_over_run": float(sum(occupancy_history) / len(occupancy_history)),
                "occupancy_min_over_run": float(min(occupancy_history)),
                "occupancy_max_over_run": float(max(occupancy_history)),
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


# =========================================================
# 8) CSV EXPORT
# =========================================================
def export_csvs(
    config: SimConfig,
    visit_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    run_output_dir: str | None = None,
):
    if run_output_dir is None:
        run_output_dir = config.output_dir

    os.makedirs(run_output_dir, exist_ok=True)

    visit_path = os.path.join(run_output_dir, "visit_log.csv")
    agg_path = os.path.join(run_output_dir, "aggregated_edges.csv")
    summary_path = os.path.join(run_output_dir, "run_summary.csv")

    visit_df.to_csv(visit_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return visit_path, agg_path, summary_path


# =========================================================
# 9) VISUALIZATION
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

    staff_nodes = [n for n, d in G.nodes(data=True) if d.get("role") in {"nurse", "doctor"}]
    labels = {n: n for n in staff_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title("Aggregated Contact Network (Undirected)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_timeseries(config: SimConfig, visit_df: pd.DataFrame, out_path: str):
    total_ticks = config.ticks_per_day * config.simulation_days
    ticks = list(range(total_ticks))
    total_counts = [0] * total_ticks
    pn_counts = [0] * total_ticks
    pd_counts = [0] * total_ticks

    if not visit_df.empty:
        for _, r in visit_df.iterrows():
            tick = int(r["tick"])
            if 0 <= tick < total_ticks:
                total_counts[tick] += 1
                tpair = edge_type(r["actor_type"], r["target_type"])
                if tpair == "NP":
                    pn_counts[tick] += 1
                elif tpair == "DP":
                    pd_counts[tick] += 1

    x_days = [t * config.dt_min / MINUTES_PER_DAY for t in ticks]

    plt.figure(figsize=(14, 5))
    plt.plot(x_days, total_counts, label="Total events", linewidth=1.2)
    plt.plot(x_days, pn_counts, label="PN events", linewidth=1.0)
    plt.plot(x_days, pd_counts, label="PD events", linewidth=1.0)
    plt.xlabel("Simulation day")
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


def export_figures(config: SimConfig, visit_df: pd.DataFrame, agg_df: pd.DataFrame, run_output_dir: str | None = None):
    if run_output_dir is None:
        run_output_dir = config.output_dir

    fig_dir = os.path.join(run_output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    network_path = os.path.join(fig_dir, "network.png")
    timeseries_path = os.path.join(fig_dir, "timeseries.png")
    degree_hist_path = os.path.join(fig_dir, "degree_hist.png")

    plot_network(config, agg_df, network_path)
    plot_timeseries(config, visit_df, timeseries_path)
    plot_degree_hist(config, agg_df, degree_hist_path)

    return network_path, timeseries_path, degree_hist_path


# =========================================================
# 10) CLI & MAIN
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

    run_output_dir = build_run_output_dir(config.output_dir, config.run_id, config.seed)

    visit_df, agg_df, summary_df = run_simulation(config)
    visit_path, agg_path, summary_path = export_csvs(config, visit_df, agg_df, summary_df, run_output_dir)
    net_path, ts_path, deg_path = export_figures(config, visit_df, agg_df, run_output_dir)

    total_events = int(summary_df.loc[0, "total_events"])
    unique_edges = int(summary_df.loc[0, "unique_edges"])

    print("\n=== Simulation finished ===")
    print(f"run_id={config.run_id} | seed={config.seed}")
    print(f"total_events={total_events}, unique_edges={unique_edges}")
    print(f"\nRun-specific output directory: {run_output_dir}")
    print("\nOutput files:")
    print(f"- {visit_path}")
    print(f"- {agg_path}")
    print(f"- {summary_path}")
    print(f"- {net_path}")
    print(f"- {ts_path}")
    print(f"- {deg_path}")


if __name__ == "__main__":
    main()