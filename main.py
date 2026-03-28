#!/usr/bin/env python3
"""
Single-file Mesa prototype: 1 ward hospital contact network generation WITH infection dynamics.

REFACTORED VERSION: More agent-centric behavior with explicit agent-level state machines.
Nurses and doctors now have their own step() methods and decide their behavior based on
time blocks and assignments. The model coordinates the environment and event logging,
but agents drive most contact generation.

MULTI-DAY EXTENSION:
- 30-day simulation
- dynamic admissions/discharges
- fixed baseline staffing
- infection dynamics included

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
    p_roommate_event_per_room_per_hour: float = 0.50
    nurse_station_random_ticks_per_day: int = 10

    output_dir: str = "outputs"

    # =========================================================
    # Infection dynamics configuration
    # =========================================================
    initial_seed_infections: int = 1
    seed_in_first_days: int = 2

    p_symptomatic: float = 0.60
    infection_fatality_ratio: float = 0.0104

    # E_lat duration
    latent_shape: float = 1.3521
    latent_scale_days: float = 2.0

    # E_inf duration (presymptomatic infectious period)
    presymptomatic_shape: float = 2.0
    presymptomatic_scale_days: float = 1.5

    # I_asym recovery
    recovery_asym_shape: float = 2.0
    recovery_asym_scale_days: float = 2.5

    # I_sym recovery
    recovery_sym_shape: float = 4.0
    recovery_sym_scale_days: float = 1.75

    # death after symptomatic onset
    death_shape: float = 4.9383
    death_scale_days: float = 3.6045

    # transmission placeholders, later to calibrate
    beta_patient_patient_per_5min: float = 0.01
    beta_hcw_to_patient_per_5min: float = 0.02
    beta_patient_to_hcw_per_5min: float = 0.02

    # relative infectiousness by stage
    e_inf_relative_infectiousness: float = 0.80
    i_asym_relative_infectiousness: float = 0.60
    i_sym_relative_infectiousness: float = 1.00

    isolation_transmission_multiplier: float = 0.30

    # transient HCW contamination
    hcw_contamination_duration_days: float = 0.5
    hand_hygiene_clearance_prob_after_patient_contact: float = 0.20


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


def days_to_ticks(days: float, dt_min: int) -> int:
    return max(1, int(round((days * 24 * 60) / dt_min)))


def sample_gamma_days(rng: random.Random, shape: float, scale_days: float) -> float:
    return max(1e-9, rng.gammavariate(shape, scale_days))


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
    Passive patient agent with multi-day stay attributes and epidemiological state.
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

        # Epidemiological state:
        # S, E_lat, E_inf, I_asym, I_sym, R, D
        self.epi_state: str = "S"
        self.is_infectious: bool = False
        self.is_symptomatic: bool = False
        self.is_detected: bool = False
        self.is_isolated: bool = False

        self.infected_by: str | None = None
        self.infection_tick: int | None = None

        self.latent_until_tick: int | None = None
        self.presymptomatic_until_tick: int | None = None
        self.recovery_tick: int | None = None
        self.death_tick: int | None = None

    def step(self):
        if not self.is_active:
            return
        self.model._update_patient_infection_state(self)


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
        
        self.contaminated_until_tick: int | None = None

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

        self.contaminated_until_tick: int | None = None        

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
        self.infection_events: list[dict] = []
        self.seed_events: list[dict] = []
        self._scheduled_seed_introductions: list[dict] = []

        self.debug_seed_contacts: list[dict] = []
        self.debug_new_infections: list[dict] = []

        self.current_tick = 0
        self.current_time_min = 0

        self._feeding_block_assignments: dict[int, dict[str, list[str]]] = {}

        # FIX 1:
        # A roommate trigger ne csak room_id + hour legyen, mert akkor
        # az egész 30 napban egyszer aktiválódik ugyanarra az órára.
        # A napi resetet is megtartjuk, de biztonságosan beletesszük a nap indexet is.
        self._room_hour_triggered: set[tuple[str, int, int]] = set()

        self._recent_contacts: dict[tuple[str, str], int] = {}
        self._random_nurse_station_ticks = self._sample_random_nurse_station_ticks()

        # Multi-day tracking
        self.total_admissions = 0
        self.total_discharges = 0
        self.daily_flow_log: list[dict] = []
        self.daily_census_history: list[int] = []

        self._init_agents()
        self._assign_patients_to_rooms_deterministic()
        self._assign_nurse_room_caseloads()
        self._assign_doctor_panels()
        self._assign_daily_feeders()

        self._schedule_initial_seed_introductions()

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

    # FIX 2:
    # Nem használt paraméterek rendbetétele és LOS konzisztencia.
    # Bevezetünk központi LOS mintavételt, hogy:
    # - az inicializált betegek
    # - és az új felvételek
    # ugyanabból a deklarált logikából kapjanak tartózkodási időt.
    def _sample_los_days(self, distribution_name: str | None = None) -> int:
        dist = distribution_name or self.config.los_distribution

        if dist == "fixed":
            return max(1, int(round(self.config.mean_los_days)))

        if dist == "discrete_uniform_1_8":
            return self.rng.randint(1, 8)

        if dist == "exponential":
            mean = max(1e-9, float(self.config.mean_los_days))
            sampled = int(round(self.rng.expovariate(1.0 / mean)))
            return max(1, sampled)

        # fallback: konzervatív viselkedés
        return max(1, int(round(self.config.mean_los_days)))

    def _assign_patients_to_rooms_deterministic(self):
        initial_n = self.config.initial_patient_count
        active_patients = self.patients[:initial_n]

        for p in active_patients:
            p.is_active = True
            p.admission_day = 0

            # FIX 3:
            # korábban hardcode randint(1, 8) volt.
            # most a deklarált initial_remaining_los_distribution paramétert használjuk.
            p.remaining_los_days = self._sample_los_days(self.config.initial_remaining_los_distribution)

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

        # FIX 4:
        # korábban fix 8 nap volt, most a deklarált LOS logikát használjuk.
        patient.remaining_los_days = self._sample_los_days(self.config.los_distribution)

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
        day_idx = self.get_current_day()
        admissions_today = 0
        discharges_today = 0

        active_patients = [p for p in self.patients if p.is_active]

        for patient in active_patients:
            if patient.remaining_los_days is not None:
                patient.remaining_los_days -= 1

        for patient in list(active_patients):
            if patient.remaining_los_days is not None and patient.remaining_los_days <= 0:
                self._discharge_patient(patient)
                discharges_today += 1        
        

         # FIX 5:
        # Stochastic admission szabály a target occupancy körüli visszatöltéshez.
        current_census = self.get_current_patient_count()

        target_gap = (self.config.target_bed_occupancy * self.config.ward_capacity) - current_census
        expected_admissions = max(0.0, target_gap)

        daily_admissions = self._sample_poisson(expected_admissions)

        available_beds = self.config.ward_capacity - current_census
        daily_admissions = min(daily_admissions, available_beds)

        inactive_pool = self._get_inactive_patients_pool()

        for patient in inactive_pool:
            if daily_admissions <= 0:
                break
            room_id = self._find_first_available_bed()
            if room_id is None:
                break
            self._admit_patient(patient, room_id)
            admissions_today += 1
            daily_admissions -= 1
                

        # FIX 6:
        # napi reset a roommate triggerhez
        self._room_hour_triggered.clear()

        self._refresh_assignments_after_census_change()

        census_end_of_day = self.get_current_patient_count()
        self.daily_census_history.append(census_end_of_day)

        self.daily_flow_log.append(
            {
                "day": int(day_idx),
                "admissions": int(admissions_today),
                "discharges": int(discharges_today),
                "census_end_of_day": int(census_end_of_day),
                "occupancy_end_of_day": float(census_end_of_day / self.config.ward_capacity),
            }
        )        
        
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

        # FIX 7:
        # egységes abszolút időkezelés
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
        if (
            108 <= tick <= 1260
            and (actor_id == "patient_53" or target_id == "patient_53")
        ):
            self.debug_seed_contacts.append(
                {
                    "tick": tick,
                    "time_str": event["time_str"],
                    "actor_id": actor_id,
                    "actor_type": event["actor_type"],
                    "target_id": target_id,
                    "target_type": event["target_type"],
                    "event_type": event_type,
                    "duration_min": duration_min,
                }
            )

        self._attempt_transmission_from_contact(event)

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
    # Infection dynamics
    # =========================================================
    def _schedule_initial_seed_introductions(self):
        seed_patient_id = "patient_53"
        seed_tick = 9 * 60 // self.config.dt_min   # 09:00, nappali aktivitási időben

        self._scheduled_seed_introductions = [
            {
                "patient_id": seed_patient_id,
                "seed_tick": seed_tick,
                "seed_state": "I_asym",
            }
        ]

    def _apply_scheduled_seed_introductions(self):
        if not self._scheduled_seed_introductions:
            return

        remaining = []
        for item in self._scheduled_seed_introductions:
            if item["seed_tick"] != self.current_tick:
                remaining.append(item)
                continue

            patient = self.agent_index.get(item["patient_id"])
            if isinstance(patient, PatientAgent) and patient.is_active and patient.epi_state == "S":
                self._force_seed_patient_as_e_inf(patient)

        self._scheduled_seed_introductions = remaining

    def _force_seed_patient_as_e_inf(self, patient: PatientAgent):
        patient.epi_state = "I_asym"
        patient.is_infectious = True
        patient.is_symptomatic = False
        patient.is_detected = False
        patient.is_isolated = False

        patient.infected_by = "seed"
        patient.infection_tick = self.current_tick
        patient.latent_until_tick = None
        patient.presymptomatic_until_tick = None

        recovery_days = max(
            4.0,
            sample_gamma_days(
                self.rng,
                self.config.recovery_asym_shape,
                self.config.recovery_asym_scale_days,
            )
        )
        patient.recovery_tick = self.current_tick + days_to_ticks(
            recovery_days,
            self.config.dt_min,
        )
        patient.death_tick = None

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": tick_to_time_min(self.current_tick, self.config.dt_min),
            "time_str": minutes_to_hhmm(tick_to_time_min(self.current_tick, self.config.dt_min)),
            "patient_id": patient.unique_id,
            "source_id": "seed",
            "source_type": "seed",
            "event_type": "seed_introduction",
            "new_state": "I_asym",
        }
        self.infection_events.append(event)
        self.seed_events.append(event)

    def _is_staff_agent(self, agent: BaseHospitalAgent) -> bool:
        return isinstance(agent, (NurseAgent, DoctorAgent))

    def _is_patient_susceptible(self, agent: BaseHospitalAgent) -> bool:
        return isinstance(agent, PatientAgent) and agent.is_active and agent.epi_state == "S"

    def _is_patient_infectious(self, agent: BaseHospitalAgent) -> bool:
        return isinstance(agent, PatientAgent) and agent.is_active and agent.epi_state in {"E_inf", "I_asym", "I_sym"}

    def _is_staff_contaminated(self, staff: BaseHospitalAgent) -> bool:
        if not self._is_staff_agent(staff):
            return False
        return staff.contaminated_until_tick is not None and self.current_tick < staff.contaminated_until_tick

    def _clear_staff_contamination(self, staff: BaseHospitalAgent):
        if self._is_staff_agent(staff):
            staff.contaminated_until_tick = None

    def _contaminate_staff(self, staff: BaseHospitalAgent):
        if not self._is_staff_agent(staff):
            return
        duration_ticks = days_to_ticks(self.config.hcw_contamination_duration_days, self.config.dt_min)
        staff.contaminated_until_tick = self.current_tick + duration_ticks

    def _get_infectiousness_multiplier(self, patient: PatientAgent) -> float:
        if patient.epi_state == "E_inf":
            return self.config.e_inf_relative_infectiousness
        if patient.epi_state == "I_asym":
            return self.config.i_asym_relative_infectiousness
        if patient.epi_state == "I_sym":
            return self.config.i_sym_relative_infectiousness
        return 0.0

    def _get_effective_transmission_prob(
        self,
        base_beta_per_5min: float,
        duration_min: int,
        infectiousness_multiplier: float = 1.0,
        isolation_multiplier: float = 1.0,
    ) -> float:
        n_units = max(1.0, duration_min / self.config.dt_min)
        beta_eff = base_beta_per_5min * infectiousness_multiplier * isolation_multiplier
        beta_eff = min(max(beta_eff, 0.0), 1.0)
        return 1.0 - ((1.0 - beta_eff) ** n_units)

    def _infect_patient_from_contact(
        self,
        patient: PatientAgent,
        source_id: str,
        source_type: str,
        event_type: str,
    ):
        if not patient.is_active:
            return
        if patient.epi_state != "S":
            return
        self.debug_new_infections.append(
            {
                "tick": self.current_tick,
                "patient_id": patient.unique_id,
                "source_id": source_id,
                "source_type": source_type,
                "event_type": event_type,
            }
        )
        patient.epi_state = "E_lat"
        patient.is_infectious = False
        patient.is_symptomatic = False
        patient.is_detected = False
        patient.is_isolated = False

        patient.infected_by = source_id
        patient.infection_tick = self.current_tick

        latent_days = sample_gamma_days(
            self.rng,
            self.config.latent_shape,
            self.config.latent_scale_days,
        )
        patient.latent_until_tick = self.current_tick + days_to_ticks(latent_days, self.config.dt_min)
        patient.presymptomatic_until_tick = None
        patient.recovery_tick = None
        patient.death_tick = None

        self.infection_events.append(
            {
                "run_id": self.config.run_id,
                "tick": self.current_tick,
                "day": self.get_current_day(),
                "time_min": tick_to_time_min(self.current_tick, self.config.dt_min),
                "time_str": minutes_to_hhmm(tick_to_time_min(self.current_tick, self.config.dt_min)),
                "patient_id": patient.unique_id,
                "source_id": source_id,
                "source_type": source_type,
                "event_type": event_type,
                "new_state": "E_lat",
            }
        )

    def _update_patient_infection_state(self, patient: PatientAgent):
        tick = self.current_tick

        if patient.epi_state == "E_lat":
            if patient.latent_until_tick is not None and tick >= patient.latent_until_tick:
                self._progress_e_lat_to_e_inf(patient)
                return

        if patient.epi_state == "E_inf":
            if patient.presymptomatic_until_tick is not None and tick >= patient.presymptomatic_until_tick:
                self._progress_e_inf_to_i_state(patient)
                return

        if patient.epi_state in {"I_asym", "I_sym"}:
            if patient.death_tick is not None and tick >= patient.death_tick:
                self._process_patient_death(patient)
                return

            if patient.recovery_tick is not None and tick >= patient.recovery_tick:
                self._process_patient_recovery(patient)
                return

    def _progress_e_lat_to_e_inf(self, patient: PatientAgent):
        if not patient.is_active or patient.epi_state != "E_lat":
            return

        patient.epi_state = "E_inf"
        patient.is_infectious = True
        patient.is_symptomatic = False
        patient.is_detected = False
        patient.is_isolated = False
        patient.latent_until_tick = None

        presymptomatic_days = sample_gamma_days(
            self.rng,
            self.config.presymptomatic_shape,
            self.config.presymptomatic_scale_days,
        )
        patient.presymptomatic_until_tick = self.current_tick + days_to_ticks(
            presymptomatic_days,
            self.config.dt_min,
        )

        self.infection_events.append(
            {
                "run_id": self.config.run_id,
                "tick": self.current_tick,
                "day": self.get_current_day(),
                "time_min": tick_to_time_min(self.current_tick, self.config.dt_min),
                "time_str": minutes_to_hhmm(tick_to_time_min(self.current_tick, self.config.dt_min)),
                "patient_id": patient.unique_id,
                "source_id": patient.infected_by,
                "source_type": "progression",
                "event_type": "state_transition",
                "new_state": "E_inf",
            }
        )

    def _progress_e_inf_to_i_state(self, patient: PatientAgent):
        if not patient.is_active or patient.epi_state != "E_inf":
            return

        patient.presymptomatic_until_tick = None

        symptomatic = self.rng.random() < self.config.p_symptomatic
        if symptomatic:
            patient.epi_state = "I_sym"
            patient.is_infectious = True
            patient.is_symptomatic = True
            patient.is_detected = False
            patient.is_isolated = False

            recovery_days = max(
                2.0,
                sample_gamma_days(
                    self.rng,
                    self.config.recovery_asym_shape,
                    self.config.recovery_asym_scale_days,
                )
            )
            patient.recovery_tick = self.current_tick + days_to_ticks(recovery_days, self.config.dt_min)

            if self.rng.random() < self.config.infection_fatality_ratio:
                death_days = sample_gamma_days(
                    self.rng,
                    self.config.death_shape,
                    self.config.death_scale_days,
                )
                patient.death_tick = self.current_tick + days_to_ticks(death_days, self.config.dt_min)
            else:
                patient.death_tick = None

            new_state = "I_sym"
        else:
            patient.epi_state = "I_asym"
            patient.is_infectious = True
            patient.is_symptomatic = False
            patient.is_detected = False
            patient.is_isolated = False

            recovery_days = max(
                3.0,
                sample_gamma_days(
                    self.rng,
                    self.config.recovery_sym_shape,
                    self.config.recovery_sym_scale_days,
                )
            )
            patient.recovery_tick = self.current_tick + days_to_ticks(recovery_days, self.config.dt_min)
            patient.death_tick = None
            new_state = "I_asym"

        self.infection_events.append(
            {
                "run_id": self.config.run_id,
                "tick": self.current_tick,
                "day": self.get_current_day(),
                "time_min": tick_to_time_min(self.current_tick, self.config.dt_min),
                "time_str": minutes_to_hhmm(tick_to_time_min(self.current_tick, self.config.dt_min)),
                "patient_id": patient.unique_id,
                "source_id": patient.infected_by,
                "source_type": "progression",
                "event_type": "state_transition",
                "new_state": new_state,
            }
        )

    def _process_patient_recovery(self, patient: PatientAgent):
        if not patient.is_active:
            return

        patient.epi_state = "R"
        patient.is_infectious = False
        patient.is_symptomatic = False
        patient.is_detected = False
        patient.is_isolated = False
        patient.latent_until_tick = None
        patient.presymptomatic_until_tick = None
        patient.recovery_tick = None
        patient.death_tick = None

        self.infection_events.append(
            {
                "run_id": self.config.run_id,
                "tick": self.current_tick,
                "day": self.get_current_day(),
                "time_min": tick_to_time_min(self.current_tick, self.config.dt_min),
                "time_str": minutes_to_hhmm(tick_to_time_min(self.current_tick, self.config.dt_min)),
                "patient_id": patient.unique_id,
                "source_id": patient.infected_by,
                "source_type": "progression",
                "event_type": "state_transition",
                "new_state": "R",
            }
        )

    def _process_patient_death(self, patient: PatientAgent):
        if not patient.is_active:
            return

        patient.epi_state = "D"
        patient.is_infectious = False
        patient.is_symptomatic = False
        patient.is_detected = False
        patient.is_isolated = False

        self.infection_events.append(
            {
                "run_id": self.config.run_id,
                "tick": self.current_tick,
                "day": self.get_current_day(),
                "time_min": tick_to_time_min(self.current_tick, self.config.dt_min),
                "time_str": minutes_to_hhmm(tick_to_time_min(self.current_tick, self.config.dt_min)),
                "patient_id": patient.unique_id,
                "source_id": patient.infected_by,
                "source_type": "progression",
                "event_type": "state_transition",
                "new_state": "D",
            }
        )

        self._discharge_patient(patient)

    def _attempt_transmission_from_contact(self, event: dict):
        print("TRANSMISSION CALLED", event["event_type"])
        actor = self.agent_index[event["actor_id"]]
        target = self.agent_index[event["target_id"]]
        duration_min = int(event["duration_min"])
        event_type = event["event_type"]
        if (
            57 <= self.current_tick <= 633
            and (
                event["actor_id"] == "patient_53"
                or event["target_id"] == "patient_53"
            )
        ):
            print(
                "INFECTIOUS WINDOW DEBUG |",
                f"tick={self.current_tick}",
                f"event={event_type}",
                f"actor={event['actor_id']}",
                f"actor_state={getattr(actor, 'epi_state', 'NA')}",
                f"actor_inf={getattr(actor, 'is_infectious', 'NA')}",
                f"target={event['target_id']}",
                f"target_state={getattr(target, 'epi_state', 'NA')}",
                f"target_inf={getattr(target, 'is_infectious', 'NA')}",
                f"duration={duration_min}",
            )

        # patient-patient direct transmission
        if isinstance(actor, PatientAgent) and isinstance(target, PatientAgent):
            if self._is_patient_infectious(actor) and self._is_patient_susceptible(target):
                infectiousness = self._get_infectiousness_multiplier(actor)
                isolation_multiplier = (
                    self.config.isolation_transmission_multiplier
                    if (actor.is_isolated or target.is_isolated)
                    else 1.0
                )
                p = self._get_effective_transmission_prob(
                    self.config.beta_patient_patient_per_5min,
                    duration_min,
                    infectiousness_multiplier=infectiousness,
                    isolation_multiplier=isolation_multiplier,
                )
                if actor.unique_id == "patient_53" or target.unique_id == "patient_53":
                    print(
                        "PP CHECK |",
                        f"tick={self.current_tick}",
                        f"actor={actor.unique_id}",
                        f"actor_state={actor.epi_state}",
                        f"actor_inf={actor.is_infectious}",
                        f"target={target.unique_id}",
                        f"target_state={target.epi_state}",
                        f"target_inf={target.is_infectious}",
                        f"duration={duration_min}",
                        f"p={p:.4f}",
                    )
                if self.rng.random() < p:
                    self._infect_patient_from_contact(
                        target,
                        source_id=actor.unique_id,
                        source_type=actor.agent_type,
                        event_type=event_type,
                    )

            elif self._is_patient_infectious(target) and self._is_patient_susceptible(actor):
                infectiousness = self._get_infectiousness_multiplier(target)
                isolation_multiplier = (
                    self.config.isolation_transmission_multiplier
                    if (actor.is_isolated or target.is_isolated)
                    else 1.0
                )
                p = self._get_effective_transmission_prob(
                    self.config.beta_patient_patient_per_5min,
                    duration_min,
                    infectiousness_multiplier=infectiousness,
                    isolation_multiplier=isolation_multiplier,
                )
                if target.unique_id == "patient_53" or actor.unique_id == "patient_53":
                    print(
                        "PP CHECK |",
                        f"tick={self.current_tick}",
                        f"actor={actor.unique_id}",
                        f"actor_state={actor.epi_state}",
                        f"actor_inf={actor.is_infectious}",
                        f"target={target.unique_id}",
                        f"target_state={target.epi_state}",
                        f"target_inf={target.is_infectious}",
                        f"duration={duration_min}",
                        f"p={p:.4f}",
                    )
                if self.rng.random() < p:
                    self._infect_patient_from_contact(
                        actor,
                        source_id=target.unique_id,
                        source_type=target.agent_type,
                        event_type=event_type,
                    )
            return

        # staff-patient contact via transient staff contamination
        if self._is_staff_agent(actor) and isinstance(target, PatientAgent):
            staff = actor
            patient = target

            # infectious patient contaminates staff
            if self._is_patient_infectious(patient):
                infectiousness = self._get_infectiousness_multiplier(patient)
                isolation_multiplier = self.config.isolation_transmission_multiplier if patient.is_isolated else 1.0
                p_contam = self._get_effective_transmission_prob(
                    self.config.beta_patient_to_hcw_per_5min,
                    duration_min,
                    infectiousness_multiplier=infectiousness,
                    isolation_multiplier=isolation_multiplier,
                )
                if self.rng.random() < p_contam:
                    self._contaminate_staff(staff)

            # contaminated staff infects susceptible patient
            elif self._is_staff_contaminated(staff) and self._is_patient_susceptible(patient):
                isolation_multiplier = self.config.isolation_transmission_multiplier if patient.is_isolated else 1.0
                p_trans = self._get_effective_transmission_prob(
                    self.config.beta_hcw_to_patient_per_5min,
                    duration_min,
                    infectiousness_multiplier=1.0,
                    isolation_multiplier=isolation_multiplier,
                )
                if self.rng.random() < p_trans:
                    self._infect_patient_from_contact(
                        patient,
                        source_id=staff.unique_id,
                        source_type=staff.agent_type,
                        event_type=event_type,
                    )

            # hand hygiene can clear staff contamination after patient contact
            if self._is_staff_contaminated(staff):
                if self.rng.random() < self.config.hand_hygiene_clearance_prob_after_patient_contact:
                    self._clear_staff_contamination(staff)


    # =========================================================
    # Model-level event generation
    # =========================================================
    def _generate_roommate_events(self, tick: int, time_min: int):
        for room_id, occupants in self.room_occupants.items():
            active_occupants = [pid for pid in occupants if self.is_patient_active(pid)]

            if len(active_occupants) < 2:
                continue

            for i in range(len(active_occupants)):
                for j in range(i + 1, len(active_occupants)):
                    p1 = active_occupants[i]
                    p2 = active_occupants[j]

                    self.record_contact(
                        tick=tick,
                        actor_id=p1,
                        target_id=p2,
                        event_type="roommate",
                        duration_min=self.config.dt_min,
                    )

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

        self._apply_scheduled_seed_introductions()   

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

def run_simulation(config: SimConfig) -> tuple[HospitalContactModel, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return model, visit_df, agg_df, summary_df


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
                "final_S": int(sum(1 for p in model.patients if p.epi_state == "S")),
                "final_E_lat": int(sum(1 for p in model.patients if p.epi_state == "E_lat")),
                "final_E_inf": int(sum(1 for p in model.patients if p.epi_state == "E_inf")),
                "final_I_asym": int(sum(1 for p in model.patients if p.epi_state == "I_asym")),
                "final_I_sym": int(sum(1 for p in model.patients if p.epi_state == "I_sym")),
                "final_R": int(sum(1 for p in model.patients if p.epi_state == "R")),
                "final_D": int(sum(1 for p in model.patients if p.epi_state == "D")),
                "total_infection_events": int(len([e for e in model.infection_events if e["new_state"] in {"E_lat", "E_inf"}])),
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

def export_infection_csv(model: HospitalContactModel, run_output_dir: str) -> str:
    infection_path = os.path.join(run_output_dir, "infection_log.csv")
    infection_df = pd.DataFrame(model.infection_events)
    infection_df.to_csv(infection_path, index=False)
    return infection_path

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
# 9/B) OUTPUT ANALYSIS
# =========================================================
def _normalize_role_pair(actor_type: str, target_type: str) -> str:
    pair = tuple(sorted([actor_type, target_type]))
    mapping = {
        ("patient", "patient"): "PP",
        ("nurse", "patient"): "NP",
        ("doctor", "patient"): "DP",
        ("nurse", "nurse"): "NN",
        ("doctor", "nurse"): "DN",
        ("doctor", "doctor"): "DD",
    }
    return mapping[pair]


def build_daily_timeseries_dataset(config: SimConfig, visit_df: pd.DataFrame) -> pd.DataFrame:
    if visit_df.empty:
        return pd.DataFrame(
            columns=[
                "day",
                "total_events",
                "PP",
                "NP",
                "DP",
                "NN",
                "DN",
                "DD",
                "unique_patients",
                "unique_staff",
                "unique_all_agents",
            ]
        )

    df = visit_df.copy()

    if "day" not in df.columns:
        df["day"] = (df["tick"] // config.ticks_per_day).astype(int)

    df["role_pair"] = df.apply(
        lambda r: _normalize_role_pair(r["actor_type"], r["target_type"]),
        axis=1,
    )

    daily_total = df.groupby("day").size().rename("total_events")

    daily_pairs = (
        df.groupby(["day", "role_pair"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["PP", "NP", "DP", "NN", "DN", "DD"], fill_value=0)
    )

    patient_actor = df.loc[df["actor_type"] == "patient", ["day", "actor_id"]].rename(
        columns={"actor_id": "agent_id"}
    )
    patient_target = df.loc[df["target_type"] == "patient", ["day", "target_id"]].rename(
        columns={"target_id": "agent_id"}
    )
    daily_unique_patients = (
        pd.concat([patient_actor, patient_target], ignore_index=True)
        .drop_duplicates()
        .groupby("day")["agent_id"]
        .nunique()
        .rename("unique_patients")
    )

    staff_actor = df.loc[df["actor_type"].isin(["nurse", "doctor"]), ["day", "actor_id"]].rename(
        columns={"actor_id": "agent_id"}
    )
    staff_target = df.loc[df["target_type"].isin(["nurse", "doctor"]), ["day", "target_id"]].rename(
        columns={"target_id": "agent_id"}
    )
    daily_unique_staff = (
        pd.concat([staff_actor, staff_target], ignore_index=True)
        .drop_duplicates()
        .groupby("day")["agent_id"]
        .nunique()
        .rename("unique_staff")
    )

    all_actor = df[["day", "actor_id"]].rename(columns={"actor_id": "agent_id"})
    all_target = df[["day", "target_id"]].rename(columns={"target_id": "agent_id"})
    daily_unique_all = (
        pd.concat([all_actor, all_target], ignore_index=True)
        .drop_duplicates()
        .groupby("day")["agent_id"]
        .nunique()
        .rename("unique_all_agents")
    )

    ts_df = pd.concat(
        [
            daily_total,
            daily_pairs,
            daily_unique_patients,
            daily_unique_staff,
            daily_unique_all,
        ],
        axis=1,
    ).fillna(0).reset_index()

    int_cols = [
        "day",
        "total_events",
        "PP",
        "NP",
        "DP",
        "NN",
        "DN",
        "DD",
        "unique_patients",
        "unique_staff",
        "unique_all_agents",
    ]
    for c in int_cols:
        if c in ts_df.columns:
            ts_df[c] = ts_df[c].astype(int)

    return ts_df


def build_role_pair_tables(visit_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if visit_df.empty:
        summary = pd.DataFrame(columns=["role_pair", "count", "ratio"])
        matrix_counts = pd.DataFrame(
            0,
            index=["patient", "nurse", "doctor"],
            columns=["patient", "nurse", "doctor"],
        )
        matrix_ratios = matrix_counts.astype(float)
        return summary, matrix_counts, matrix_ratios

    df = visit_df.copy()
    df["role_pair"] = df.apply(
        lambda r: _normalize_role_pair(r["actor_type"], r["target_type"]),
        axis=1,
    )

    pair_counts = (
        df["role_pair"]
        .value_counts()
        .reindex(["PP", "NP", "DP", "NN", "DN", "DD"], fill_value=0)
    )

    summary = pair_counts.rename_axis("role_pair").reset_index(name="count")
    total = summary["count"].sum()
    summary["ratio"] = summary["count"] / total if total > 0 else 0.0

    matrix_counts = pd.DataFrame(
        0,
        index=["patient", "nurse", "doctor"],
        columns=["patient", "nurse", "doctor"],
    )

    mapping = {
        "PP": ("patient", "patient"),
        "NP": ("patient", "nurse"),
        "DP": ("patient", "doctor"),
        "NN": ("nurse", "nurse"),
        "DN": ("nurse", "doctor"),
        "DD": ("doctor", "doctor"),
    }

    for _, row in summary.iterrows():
        rp = row["role_pair"]
        count = int(row["count"])
        a, b = mapping[rp]
        matrix_counts.loc[a, b] = count
        matrix_counts.loc[b, a] = count

    matrix_ratios = matrix_counts / total if total > 0 else matrix_counts.astype(float)

    return summary, matrix_counts, matrix_ratios


def build_degree_tables(agg_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if agg_df.empty:
        node_df = pd.DataFrame(columns=["node_id", "role", "degree", "weighted_degree"])
        role_df = pd.DataFrame(
            columns=["role", "n_nodes", "mean_degree", "median_degree", "std_degree", "min_degree", "max_degree",
                     "mean_weighted_degree", "median_weighted_degree", "std_weighted_degree",
                     "min_weighted_degree", "max_weighted_degree"]
        )
        return node_df, role_df

    left = agg_df[["u_id", "u_type", "total_contact_count"]].rename(
        columns={"u_id": "node_id", "u_type": "role"}
    )
    right = agg_df[["v_id", "v_type", "total_contact_count"]].rename(
        columns={"v_id": "node_id", "v_type": "role"}
    )

    long_df = pd.concat([left, right], ignore_index=True)

    node_df = (
        long_df.groupby(["node_id", "role"], as_index=False)
        .agg(
            degree=("total_contact_count", "size"),
            weighted_degree=("total_contact_count", "sum"),
        )
        .sort_values(["role", "weighted_degree", "degree"], ascending=[True, False, False])
        .reset_index(drop=True)
    )

    role_df = (
        node_df.groupby("role", as_index=False)
        .agg(
            n_nodes=("node_id", "count"),
            mean_degree=("degree", "mean"),
            median_degree=("degree", "median"),
            std_degree=("degree", "std"),
            min_degree=("degree", "min"),
            max_degree=("degree", "max"),
            mean_weighted_degree=("weighted_degree", "mean"),
            median_weighted_degree=("weighted_degree", "median"),
            std_weighted_degree=("weighted_degree", "std"),
            min_weighted_degree=("weighted_degree", "min"),
            max_weighted_degree=("weighted_degree", "max"),
        )
    )

    role_df = role_df.fillna(0)

    return node_df, role_df

def build_daily_flow_dataframe(model: HospitalContactModel) -> pd.DataFrame:
    if not model.daily_flow_log:
        return pd.DataFrame(
            columns=[
                "day",
                "admissions",
                "discharges",
                "census_end_of_day",
                "occupancy_end_of_day",
            ]
        )

    df = pd.DataFrame(model.daily_flow_log).sort_values("day").reset_index(drop=True)
    return df

def plot_daily_events_analysis(ts_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(14, 6))

    if ts_df.empty:
        plt.title("Daily events (empty)")
        plt.xlabel("Day")
        plt.ylabel("Events")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    plt.plot(ts_df["day"], ts_df["total_events"], label="Total", linewidth=1.8)
    plt.plot(ts_df["day"], ts_df["PP"], label="PP", linewidth=1.0)
    plt.plot(ts_df["day"], ts_df["NP"], label="NP", linewidth=1.0)
    plt.plot(ts_df["day"], ts_df["DP"], label="DP", linewidth=1.0)

    plt.xlabel("Day")
    plt.ylabel("Events")
    plt.title("Daily contact events")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_daily_unique_patients(ts_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(14, 5))

    if ts_df.empty:
        plt.title("Daily unique patients (empty)")
        plt.xlabel("Day")
        plt.ylabel("Unique patients")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    plt.plot(ts_df["day"], ts_df["unique_patients"], linewidth=1.5)
    plt.xlabel("Day")
    plt.ylabel("Unique patients with any contact")
    plt.title("Daily unique patients")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_role_pair_bar(role_pair_summary_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10, 5))

    if role_pair_summary_df.empty:
        plt.title("Role-pair counts (empty)")
        plt.xlabel("Role pair")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    plt.bar(role_pair_summary_df["role_pair"], role_pair_summary_df["count"])
    plt.xlabel("Role pair")
    plt.ylabel("Count")
    plt.title("Role-pair contact counts")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_edge_weight_histogram(agg_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10, 5))

    if agg_df.empty:
        plt.title("Edge weight distribution (empty)")
        plt.xlabel("Total contact count")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    plt.hist(agg_df["total_contact_count"], bins=20, alpha=0.8)
    plt.xlabel("Edge weight (total contact count)")
    plt.ylabel("Frequency")
    plt.title("Edge weight distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_daily_flow(daily_flow_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(14, 6))

    if daily_flow_df.empty:
        plt.title("Daily admissions, discharges, and census (empty)")
        plt.xlabel("Day")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    plt.plot(
        daily_flow_df["day"],
        daily_flow_df["census_end_of_day"],
        label="Census end of day",
        linewidth=1.8,
    )
    plt.plot(
        daily_flow_df["day"],
        daily_flow_df["admissions"],
        label="Admissions",
        linewidth=1.2,
    )
    plt.plot(
        daily_flow_df["day"],
        daily_flow_df["discharges"],
        label="Discharges",
        linewidth=1.2,
    )

    plt.xlabel("Day")
    plt.ylabel("Count")
    plt.title("Daily admissions, discharges, and census")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def export_analysis_outputs(
    config: SimConfig,
    model: HospitalContactModel,
    visit_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    run_output_dir: str,
):
    analysis_dir = os.path.join(run_output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    ts_df = build_daily_timeseries_dataset(config, visit_df)
    role_pair_summary_df, role_pair_matrix_counts_df, role_pair_matrix_ratios_df = build_role_pair_tables(visit_df)
    degree_node_df, degree_role_df = build_degree_tables(agg_df)
    daily_flow_df = build_daily_flow_dataframe(model)    
    
    ts_path = os.path.join(analysis_dir, "timeseries_daily.csv")
    role_pair_summary_path = os.path.join(analysis_dir, "role_pair_summary.csv")
    role_pair_counts_path = os.path.join(analysis_dir, "role_pair_matrix_counts.csv")
    role_pair_ratios_path = os.path.join(analysis_dir, "role_pair_matrix_ratios.csv")
    degree_node_path = os.path.join(analysis_dir, "degree_summary_by_node.csv")
    degree_role_path = os.path.join(analysis_dir, "degree_summary_by_role.csv")
    daily_flow_path = os.path.join(analysis_dir, "daily_flow.csv")

    ts_df.to_csv(ts_path, index=False)
    role_pair_summary_df.to_csv(role_pair_summary_path, index=False)
    role_pair_matrix_counts_df.to_csv(role_pair_counts_path, index=True)
    role_pair_matrix_ratios_df.to_csv(role_pair_ratios_path, index=True)
    degree_node_df.to_csv(degree_node_path, index=False)
    degree_role_df.to_csv(degree_role_path, index=False)
    daily_flow_df.to_csv(daily_flow_path, index=False)

    daily_events_fig_path = os.path.join(analysis_dir, "daily_events.png")
    daily_unique_patients_fig_path = os.path.join(analysis_dir, "daily_unique_patients.png")
    role_pair_bar_fig_path = os.path.join(analysis_dir, "role_pair_bar.png")
    edge_weight_hist_fig_path = os.path.join(analysis_dir, "edge_weight_hist.png")
    daily_flow_fig_path = os.path.join(analysis_dir, "daily_flow.png")

    plot_daily_events_analysis(ts_df, daily_events_fig_path)
    plot_daily_unique_patients(ts_df, daily_unique_patients_fig_path)
    plot_role_pair_bar(role_pair_summary_df, role_pair_bar_fig_path)
    plot_edge_weight_histogram(agg_df, edge_weight_hist_fig_path)
    plot_daily_flow(daily_flow_df, daily_flow_fig_path)

    return {
        "analysis_dir": analysis_dir,
        "timeseries_daily_csv": ts_path,
        "role_pair_summary_csv": role_pair_summary_path,
        "role_pair_matrix_counts_csv": role_pair_counts_path,
        "role_pair_matrix_ratios_csv": role_pair_ratios_path,
        "degree_summary_by_node_csv": degree_node_path,
        "degree_summary_by_role_csv": degree_role_path,
        "daily_flow_csv": daily_flow_path,
        "daily_events_png": daily_events_fig_path,
        "daily_unique_patients_png": daily_unique_patients_fig_path,
        "role_pair_bar_png": role_pair_bar_fig_path,
        "edge_weight_hist_png": edge_weight_hist_fig_path,
        "daily_flow_png": daily_flow_fig_path,
    }

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

    model, visit_df, agg_df, summary_df = run_simulation(config)
    visit_path, agg_path, summary_path = export_csvs(config, visit_df, agg_df, summary_df, run_output_dir)
    infection_path = export_infection_csv(model, run_output_dir)
    net_path, ts_path, deg_path = export_figures(config, visit_df, agg_df, run_output_dir)

    analysis_paths = export_analysis_outputs(
        config=config,
        model=model,
        visit_df=visit_df,
        agg_df=agg_df,
        summary_df=summary_df,
        run_output_dir=run_output_dir,
    )

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
    print(f"- {infection_path}")
    print(f"- {net_path}")
    print(f"- {ts_path}")
    print(f"- {deg_path}")
    print("\nAnalysis files:")
    print(f"- {analysis_paths['timeseries_daily_csv']}")
    print(f"- {analysis_paths['role_pair_summary_csv']}")
    print(f"- {analysis_paths['role_pair_matrix_counts_csv']}")
    print(f"- {analysis_paths['role_pair_matrix_ratios_csv']}")
    print(f"- {analysis_paths['degree_summary_by_node_csv']}")
    print(f"- {analysis_paths['degree_summary_by_role_csv']}")
    print(f"- {analysis_paths['daily_flow_csv']}")
    print(f"- {analysis_paths['daily_events_png']}")
    print(f"- {analysis_paths['daily_unique_patients_png']}")
    print(f"- {analysis_paths['role_pair_bar_png']}")
    print(f"- {analysis_paths['edge_weight_hist_png']}")
    print(f"- {analysis_paths['daily_flow_png']}")
    print("\n=== DEBUG SUMMARY ===")
    print(f"seed-window contacts involving patient_53: {len(model.debug_seed_contacts)}")
    for row in model.debug_seed_contacts[:20]:
        print(row)

    print(f"\nnew infections created: {len(model.debug_new_infections)}")
    for row in model.debug_new_infections[:20]:
        print(row)
if __name__ == "__main__":
    main()
