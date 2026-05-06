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
- dynamic staff pool with day/night shift assignment
- infection dynamics included

Outputs include:
- visit_log.csv
- infection_log.csv
- flow_log.csv
- state_snapshot.csv
- aggregated_edges.csv
- run_summary.csv
- run_metadata.json
- daily_flow.csv
- timeseries_daily.csv
- role-pair and degree summary files
- optional figures
"""

# =========================================================
# 1) IMPORTS & CONSTANTS
# =========================================================

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field
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

ROOM_CONFIGS = {
    # 83 beds, current baseline structure: 1x1-bed, 5x2-bed, 18x4-bed
    "baseline": [1, 2, 2, 2, 2, 2] + [4] * 18,

    # 83 beds, small-room dominant structure: 21x1-bed, 31x2-bed
    "small": [1] * 21 + [2] * 31,

    # 83 beds, medium-room structure: 17x3-bed, 8x4-bed
    "medium": [3] * 17 + [4] * 8,

    # 83 beds, large-room structure: 13x5-bed, 3x6-bed
    "large": [5] * 13 + [6] * 3,
}

N_PATIENTS = sum(ROOM_CAPACITY_SPEC)  # 83
N_ROOMS = len(ROOM_CAPACITY_SPEC)     # 24


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

    room_config_name: str = "baseline"
    room_capacity_spec: list[int] = field(default_factory=lambda: list(ROOM_CONFIGS["baseline"]))

    ward_capacity: int = N_PATIENTS
    n_patients: int = N_PATIENTS
    # teljes dolgozói állomány, nem az egy időben aktív baseline
    n_nurses: int = 14
    n_doctors: int = 6
    n_rooms: int = N_ROOMS

    target_bed_occupancy: float = 0.6829
    mean_los_days: float = 7.69
    initial_patient_count: int = 57

    # Betegáramlási referenciaértékek.
    # A target_bed_occupancy országos éves átlagból becsült referencia,
    # nem napi kemény kapacitáskorlát.
    census_soft_lower: int = 50
    census_soft_upper: int = 65
    census_hard_upper: int = 83 # szükség esetén baseline 70

    los_distribution: str = "gamma"
    initial_remaining_los_distribution: str = "random_point_in_stay"


    los_gamma_shape: float = 2.0
    los_min_days: int = 1
    los_max_days: int = 30

    baseline_nurses_day: int = 7
    baseline_nurses_night: int = 5
    baseline_doctors_day: int = 4
    baseline_doctors_night: int = 1

    shift_length_hours: int = 12
    nurse_shift_times: str = "07:00-19:00"
    doctor_shift_times: str = "08:00-17:00"
    staff_rotation_mode: str = "cyclic_with_shift_pools"
    staff_symptomatic_stay_home_probability: float = 1.0  #realisztikusabb presenteeism-szcenárió 1.0 helyett 0.7  1.0 minden tünetes kiesik, 0.0 senki nem esik ki, 0,4 például 40% esik ki
    staff_off_shift_community_exposure_per_day: float = 0
    staff_symptomatic_off_duty_days: int = 10             # realisztikusabb presenteeism-szcenárió 10 helyett 5

    nurse_rounds_per_day: int = 3
    doctor_visits_per_patient_per_day: int = 1

    day_activity_share: float = 0.941
    night_activity_share: float = 0.059

    mean_nurse_patient_contact_duration_min: int = 10
    mean_doctor_patient_contact_duration_min: int = 4
    contact_duration_distribution: str = "exponential"

    admission_room_assignment_rule: str = "lowest_occupancy_ratio_then_random"
    day_boundary_rule: str = "decrement_los_then_discharge_then_admit_then_refresh_assignments"

    feeding_coverage_min: float = 0.30
    feeding_coverage_max: float = 0.50
    p_ad_hoc_tick: float = 0.20
    ad_hoc_max_events_per_tick: int = 2
    nurse_station_random_ticks_per_day: int = 10
    roommate_contact_interval_min: int = 120

    output_dir: str = "outputs"

    # =========================================================
    # Infection dynamics configuration
    # =========================================================
    initial_seed_infections: int = 1
    seed_in_first_days: int = 1
    seed_state: str = "I_asym"          # választható: "E_lat", "E_inf", "I_asym"
    seed_start_hour: int = 9            # első seedelési időablak kezdete
    seed_end_hour: int = 9              # ha ugyanaz, fix óra; ha nagyobb, órablakból mintáz
    seed_only_active_patients: bool = True

    
    p_symptomatic: float = 0.60
    infection_fatality_ratio: float = 0.0104

    # E_lat duration
    latent_shape: float = 1.3521
    latent_scale_days: float = 1.10

    # E_inf duration (presymptomatic infectious period)
    presymptomatic_shape: float = 2.0
    presymptomatic_scale_days: float = 0.90

    # I_asym recovery
    recovery_asym_shape: float = 2.0
    recovery_asym_scale_days: float = 2.5

    # I_sym recovery
    recovery_sym_shape: float = 4.0
    recovery_sym_scale_days: float = 1.75

    # death after symptomatic onset
    death_shape: float = 4.9383
    death_scale_days: float = 3.6045

    # transmission probabilities per 5-minute contact interval
    beta_patient_patient_per_5min: float = 0.004    # alternative sensitivity-analysis value: 0.015
    beta_hcw_to_patient_per_5min: float = 0.012     # alternative sensitivity-analysis value: 0.03
    beta_patient_to_hcw_per_5min: float = 0.012     # alternative sensitivity-analysis value: 0.03
                                                    # Higher PP beta values increase the cumulative exposure effect during long roommate stays.

    # relative infectiousness by stage
    e_inf_relative_infectiousness: float = 0.90
    i_asym_relative_infectiousness: float = 0.75
    i_sym_relative_infectiousness: float = 1.00

    isolation_transmission_multiplier: float = 0.30

    # staff mask use
    mask_strategy: str = "random"  # random vagy targeted_ids
    targeted_mask_hcw_ids: str = ""

    mask_compliance_hcw: float = 0.20  # alapérték; a szakirodalomban kb. 47% is szerepelhet
    mask_source_multiplier_hcw: float = 0.30
    mask_target_multiplier_hcw: float = 0.50

    beta_hcw_hcw_per_5min: float = 0.012
    hcw_relative_infectiousness: float = 0.90

    def __post_init__(self):
        if self.room_config_name not in ROOM_CONFIGS:
            raise ValueError(
                f"Unsupported room_config_name='{self.room_config_name}'. "
                f"Allowed values: {list(ROOM_CONFIGS.keys())}"
            )

        self.room_capacity_spec = list(ROOM_CONFIGS[self.room_config_name])
        self.ward_capacity = sum(self.room_capacity_spec)
        self.n_patients = self.ward_capacity
        self.n_rooms = len(self.room_capacity_spec)

        if self.initial_patient_count > self.ward_capacity:
            raise ValueError(
                f"initial_patient_count={self.initial_patient_count} exceeds "
                f"ward_capacity={self.ward_capacity}"
            )

# =========================================================
# 3) TIME UTILITIES
# =========================================================
def build_run_output_dir(base_output_dir: str, run_id: str, seed: int) -> str:
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp_str}_seed{seed}_{run_id}"
    return os.path.join(base_output_dir, run_dir_name)


def build_batch_run_output_dir(batch_output_dir: str, run_number: int) -> str:
    return os.path.join(batch_output_dir, f"run_{run_number:03d}")

def build_batch_output_dir(base_output_dir: str, base_seed: int, n_runs: int) -> str:
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    batch_dir_name = f"{timestamp_str}_batch_seed{base_seed}_n{n_runs}"
    return os.path.join(base_output_dir, batch_dir_name)


def ensure_unique_output_dir(preferred_dir: str) -> str:
    candidate_dir = preferred_dir
    suffix = 1

    while os.path.exists(candidate_dir):
        candidate_dir = f"{preferred_dir}_{suffix:02d}"
        suffix += 1

    os.makedirs(candidate_dir, exist_ok=False)
    return candidate_dir


def tick_to_time_min(tick: int, dt_min: int) -> int:
    return tick * dt_min


def minutes_to_hhmm(total_minutes: int) -> str:
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"

def minute_to_clock_string(total_minutes: int) -> str:
    return minutes_to_hhmm(total_minutes)

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

        self.is_on_shift: bool = False
        self.shift_label: str = "off"
        self.preferred_shift: str = "day"
        self.rotation_offset: int = 0
        self.days_off_until_day: int | None = None
        self.mask_compliant: bool = False

        self.epi_state: str = "S"
        self.is_infectious: bool = False
        self.is_symptomatic: bool = False
        self.is_detected: bool = False
        self.is_isolated: bool = False

        self.infection_tick: int | None = None
        self.infected_by: str | None = None

        self.latent_until_tick: int | None = None
        self.presymptomatic_until_tick: int | None = None
        self.recovery_tick: int | None = None
        self.death_tick: int | None = None

    def step(self):
        self.model._update_staff_infection_state(self)

        if not self.is_on_shift:
            self.current_state = "off_shift"
            return

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
            other_nurses = [
                n for n in self.model.nurses
                if n.unique_id != self.unique_id and n.is_on_shift
            ]
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
            on_shift_doctors = [d for d in self.model.doctors if d.is_on_shift]
            if on_shift_doctors:
                doctor = self.model.rng.choice(on_shift_doctors)
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

        self.is_on_shift: bool = False
        self.shift_label: str = "off"
        self.preferred_shift: str = "day"
        self.rotation_offset: int = 0
        self.days_off_until_day: int | None = None
        self.mask_compliant: bool = False

        self.epi_state: str = "S"
        self.is_infectious: bool = False
        self.is_symptomatic: bool = False
        self.is_detected: bool = False
        self.is_isolated: bool = False

        self.infection_tick: int | None = None
        self.infected_by: str | None = None

        self.latent_until_tick: int | None = None
        self.presymptomatic_until_tick: int | None = None
        self.recovery_tick: int | None = None
        self.death_tick: int | None = None                

    def step(self):
        self.model._update_staff_infection_state(self)

        if not self.is_on_shift:
            self.current_state = "off_shift"
            return

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
            other_doctors = [
                d for d in self.model.doctors
                if d.unique_id != self.unique_id and d.is_on_shift
            ]
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
            on_shift_nurses = [n for n in self.model.nurses if n.is_on_shift]
            if on_shift_nurses:
                nurse = self.model.rng.choice(on_shift_nurses)
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
    def __init__(
    self,
    config: SimConfig,
    run_output_dir: str | None = None,
    stream_logs: bool = False,
):
        super().__init__()
        self.config = config
        self.rng = random.Random(config.seed)

        self.room_capacity_map: dict[str, int] = self._build_room_capacity_map()
        self.room_occupants: dict[str, list[str]] = {rid: [] for rid in self.room_capacity_map}

        self.patients: list[PatientAgent] = []
        self.nurses: list[NurseAgent] = []
        self.doctors: list[DoctorAgent] = []
        self.agent_index: dict[str, BaseHospitalAgent] = {}

        self.run_output_dir = run_output_dir
        self.stream_logs = stream_logs

        self.visit_events: list[dict] = []
        self.infection_events: list[dict] = []
        self.flow_log: list[dict] = []
        self.state_snapshot_log: list[dict] = []
        self.seed_events: list[dict] = []
        self._scheduled_seed_introductions: list[dict] = []

        self.total_infection_events_counter: int = 0
        self.total_patient_infection_events_counter: int = 0
        self.total_staff_infection_events_counter: int = 0
        self.total_nurse_infection_events_counter: int = 0
        self.total_doctor_infection_events_counter: int = 0
        self.total_patient_death_events_counter: int = 0

        self._visit_csv_file = None
        self._visit_csv_writer = None
        self._infection_csv_file = None
        self._infection_csv_writer = None
        self._flow_csv_file = None
        self._flow_csv_writer = None
        self._snapshot_csv_file = None
        self._snapshot_csv_writer = None

        self._aggregated_contact_stats: dict[tuple[str, str], dict] = {}

        self.current_tick = 0
        self.current_time_min = 0

        self._feeding_block_assignments: dict[int, dict[str, list[str]]] = {}

        self._recent_contacts: dict[tuple[str, str], int] = {}
        self._random_nurse_station_ticks = self._sample_random_nurse_station_ticks()

         # Multi-day tracking
        self.total_admissions = 0
        self.total_discharges = 0
        self.daily_flow_log: list[dict] = []
        self.daily_room_pressure_log: list[dict] = []
        self.daily_census_history: list[int] = []
        self._daily_admissions_counter: dict[int, int] = {}
        self._daily_discharges_counter: dict[int, int] = {}
        self._scheduled_discharges_by_hour: dict[tuple[int, int], list[str]] = {}
        self._scheduled_admissions_by_hour: dict[tuple[int, int], list[str]] = {}

        if self.stream_logs and self.run_output_dir is not None:
            os.makedirs(self.run_output_dir, exist_ok=True)
            self._init_stream_writers()

        self._init_agents()
        self._initialize_staff_mask_compliance()
        self._assign_patients_to_rooms_deterministic()
        self._update_staff_shift_status()
        self._assign_nurse_room_caseloads()
        self._assign_doctor_panels()
        self._assign_daily_feeders()

        self._schedule_initial_seed_introductions()

        self.daily_census_history.append(self.get_current_patient_count())

    def _init_stream_writers(self):
        visit_path = os.path.join(self.run_output_dir, "visit_log.csv")
        infection_path = os.path.join(self.run_output_dir, "infection_log.csv")
        flow_path = os.path.join(self.run_output_dir, "flow_log.csv")
        snapshot_path = os.path.join(self.run_output_dir, "state_snapshot.csv")

        self._visit_csv_file = open(visit_path, "w", newline="", encoding="utf-8")
        self._visit_csv_writer = csv.DictWriter(
            self._visit_csv_file,
            fieldnames=[
                "run_id", "tick", "day", "time_min", "time_str",
                "actor_id", "actor_type", "target_id", "target_type",
                "room_id", "event_type", "duration_min",
            ],
        )
        self._visit_csv_writer.writeheader()

        self._infection_csv_file = open(infection_path, "w", newline="", encoding="utf-8")
        self._infection_csv_writer = csv.DictWriter(
            self._infection_csv_file,
            fieldnames=[
                "run_id", "tick", "day", "time_min", "time_str",
                "patient_id", "source_id", "source_type",
                "event_type", "new_state",
            ],
        )
        self._infection_csv_writer.writeheader()

        self._flow_csv_file = open(flow_path, "w", newline="", encoding="utf-8")
        self._flow_csv_writer = csv.DictWriter(
            self._flow_csv_file,
            fieldnames=[
                "run_id", "tick", "day", "hour", "global_hour",
                "event", "patient_id", "room_id", "census", "occupancy",
            ],
        )
        self._flow_csv_writer.writeheader()

        self._snapshot_csv_file = open(snapshot_path, "w", newline="", encoding="utf-8")
        self._snapshot_csv_writer = csv.DictWriter(
            self._snapshot_csv_file,
            fieldnames=[
                "run_id", "tick", "day", "hour", "global_hour",
                "S", "E_lat", "E_inf", "I_asym", "I_sym", "R", "D",
                "active_cases", "census", "occupancy",
            ],
        )
        self._snapshot_csv_writer.writeheader()

    def _write_visit_event(self, event: dict):
        if self.stream_logs and self._visit_csv_writer is not None:
            self._visit_csv_writer.writerow(event)
            self._visit_csv_file.flush()

    def _write_infection_event(self, event: dict):
        if self.stream_logs and self._infection_csv_writer is not None:
            self._infection_csv_writer.writerow(event)
            self._infection_csv_file.flush()
        else:
            self.infection_events.append(event)
    
    def _write_flow_event(self, event: dict):
        if self.stream_logs and self._flow_csv_writer is not None:
            self._flow_csv_writer.writerow(event)
            self._flow_csv_file.flush()
        else:
            self.flow_log.append(event)
    
    def _write_snapshot_event(self, event: dict):
        self.state_snapshot_log.append(event)

        if self.stream_logs and self._snapshot_csv_writer is not None:
            self._snapshot_csv_writer.writerow(event)
            self._snapshot_csv_file.flush()
    
    def _close_stream_writers(self):
        for handle_name in [
            "_visit_csv_file",
            "_infection_csv_file",
            "_flow_csv_file",
            "_snapshot_csv_file",
        ]:
            handle = getattr(self, handle_name, None)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
    
    def _update_aggregated_contact_stats(self, event: dict):
        actor_id = event["actor_id"]
        target_id = event["target_id"]
        actor_type = event["actor_type"]
        target_type = event["target_type"]
        time_min = event["time_min"]

        if actor_id <= target_id:
            u_id, u_type, v_id, v_type = actor_id, actor_type, target_id, target_type
        else:
            u_id, u_type, v_id, v_type = target_id, target_type, actor_id, actor_type

        key = (u_id, v_id)

        if key not in self._aggregated_contact_stats:
            self._aggregated_contact_stats[key] = {
                "run_id": self.config.run_id,
                "u_id": u_id,
                "u_type": u_type,
                "v_id": v_id,
                "v_type": v_type,
                "total_contact_count": 0,
                "first_time_min": time_min,
                "last_time_min": time_min,
            }

        row = self._aggregated_contact_stats[key]
        row["total_contact_count"] += 1
        row["first_time_min"] = min(row["first_time_min"], time_min)
        row["last_time_min"] = max(row["last_time_min"], time_min)

    

    def _build_room_capacity_map(self) -> dict[str, int]:
        return {
            f"room_{i}": cap
            for i, cap in enumerate(self.config.room_capacity_spec)
        }
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
        self._initialize_staff_roster_groups()

    def _initialize_staff_roster_groups(self):
        def _assign_shift_pools(
            staff_list: list[BaseHospitalAgent],
            baseline_day: int,
            baseline_night: int,
        ):
            total = len(staff_list)
            if total == 0:
                return

            if total == 1:
                staff_list[0].preferred_shift = "day"
                staff_list[0].rotation_offset = 0
                return

            day_share = baseline_day / max(1, baseline_day + baseline_night)
            n_day_pool = int(round(total * day_share))

            n_day_pool = max(1, n_day_pool)
            n_day_pool = min(total - 1, n_day_pool)

            day_pool = staff_list[:n_day_pool]
            night_pool = staff_list[n_day_pool:]

            for idx, staff in enumerate(day_pool):
                staff.preferred_shift = "day"
                staff.rotation_offset = idx

            for idx, staff in enumerate(night_pool):
                staff.preferred_shift = "night"
                staff.rotation_offset = idx

        _assign_shift_pools(
            self.nurses,
            self.config.baseline_nurses_day,
            self.config.baseline_nurses_night,
        )
        _assign_shift_pools(
            self.doctors,
            self.config.baseline_doctors_day,
            self.config.baseline_doctors_night,
        )

    def _select_rotating_staff_for_shift(
        self,
        staff_list: list[BaseHospitalAgent],
        shift_label: str,
        required_n: int,
    ) -> list[BaseHospitalAgent]:
        current_day = self.get_current_day()

        eligible = [
            s for s in staff_list
            if getattr(s, "preferred_shift", "day") == shift_label
            and (
                getattr(s, "days_off_until_day", None) is None
                or current_day >= s.days_off_until_day
            )
        ]

        if len(eligible) < required_n:
            backups = [
                s for s in staff_list
                if s not in eligible
                and (
                    getattr(s, "days_off_until_day", None) is None
                    or current_day >= s.days_off_until_day
                )
            ]
            eligible.extend(backups)

        if not eligible:
            return []

        eligible = sorted(eligible, key=lambda s: getattr(s, "rotation_offset", 0))

        offset = current_day % len(eligible)
        rotated = eligible[offset:] + eligible[:offset]

        return rotated[:min(required_n, len(rotated))]

    def _initialize_staff_mask_compliance(self):
        all_staff = [*self.nurses, *self.doctors]

        for staff in all_staff:
            staff.mask_compliant = False

        if self.config.mask_strategy == "random":
            for staff in all_staff:
                staff.mask_compliant = self.rng.random() < self.config.mask_compliance_hcw
            return

        if self.config.mask_strategy == "targeted_ids":
            targeted_ids = {
                item.strip()
                for item in self.config.targeted_mask_hcw_ids.split(",")
                if item.strip()
            }

            for staff in all_staff:
                if staff.unique_id in targeted_ids:
                    staff.mask_compliant = True
            return

        raise ValueError(
            f"Unsupported mask_strategy='{self.config.mask_strategy}'. "
            "Allowed values: 'random', 'targeted_ids'."
        )


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

        if dist == "gamma":
            mean = max(1e-9, float(self.config.mean_los_days))
            shape = max(1e-9, float(self.config.los_gamma_shape))
            scale = mean / shape

            sampled = int(round(self.rng.gammavariate(shape, scale)))
            sampled = max(self.config.los_min_days, sampled)
            sampled = min(self.config.los_max_days, sampled)
            return sampled

        return max(1, int(round(self.config.mean_los_days)))
    
    def _sample_initial_remaining_los_days(self) -> int:
        dist = self.config.initial_remaining_los_distribution

        if dist == "random_point_in_stay":
            full_los = self._sample_los_days(self.config.los_distribution)

            # A szimuláció kezdetén a beteg már valahol a teljes tartózkodási idején belül jár.
            elapsed_days = self.rng.randint(0, max(0, full_los - 1))
            remaining_los = full_los - elapsed_days

            return max(1, remaining_los)

        return self._sample_los_days(dist)

    def _assign_patients_to_rooms_deterministic(self):
        initial_n = self.config.initial_patient_count
        active_patients = self.patients[:initial_n]

        for p in active_patients:
            p.is_active = True
            p.admission_day = 0

            p.remaining_los_days = self._sample_initial_remaining_los_days()

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

        active_nurses = [n for n in self.nurses if n.is_on_shift]
        if not active_nurses:
            return

        room_ids = sorted(self.room_capacity_map.keys(), key=lambda x: int(x.split("_")[1]))
        occupied_rooms = [rid for rid in room_ids if len(self.room_occupants[rid]) > 0]

        for i, room_id in enumerate(occupied_rooms):
            nurse = active_nurses[i % len(active_nurses)]
            nurse.caseload_rooms.append(room_id)

    def _assign_doctor_panels(self):
        for doctor in self.doctors:
            doctor.panel_patients = []

        active_doctors = [d for d in self.doctors if d.is_on_shift]
        if not active_doctors:
            return

        active_patient_ids = [p.unique_id for p in self.patients if p.is_active]
        for i, pid in enumerate(active_patient_ids):
            doctor = active_doctors[i % len(active_doctors)]
            doctor.panel_patients.append(pid)

    def _assign_daily_feeders(self):
        for nurse in self.nurses:
            nurse.is_active_feeder = False

        active_nurses = [n for n in self.nurses if n.is_on_shift]
        if not active_nurses:
            return

        k = min(2, len(active_nurses))
        selected = self.rng.sample(active_nurses, k=k)
        for nurse in selected:
            nurse.is_active_feeder = True

    def _apply_staff_off_shift_community_exposure(self):
        p_daily = self.config.staff_off_shift_community_exposure_per_day
        if p_daily <= 0:
            return

        for staff in [*self.nurses, *self.doctors]:
            if getattr(staff, "epi_state", None) != "S":
                continue

            if self.rng.random() < p_daily:
                self._infect_staff_from_contact(
                    staff,
                    source_id="community",
                    source_type="external",
                    event_type="community_exposure",
                )

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

    def get_current_hour(self) -> int:
        return self.current_time_min // 60

    def get_global_hour(self) -> int:
        return self.current_tick // (60 // self.config.dt_min)
    
    def _is_day_shift_hour(self, hour: int) -> bool:
        return 7 <= hour < 19

    def _get_current_shift_label(self) -> str:
        return "day" if self._is_day_shift_hour(self.get_current_hour()) else "night"

    def _update_staff_shift_status(self):
        shift_label = self._get_current_shift_label()

        if self.get_current_hour() == 0 and self.current_time_min == 0:
            self._apply_staff_off_shift_community_exposure()

        for nurse in self.nurses:
            nurse.is_on_shift = False
            nurse.shift_label = "off"

        for doctor in self.doctors:
            doctor.is_on_shift = False
            doctor.shift_label = "off"

        if shift_label == "day":
            active_nurses = self._select_rotating_staff_for_shift(
                self.nurses,
                "day",
                self.config.baseline_nurses_day,
            )
            active_doctors = self._select_rotating_staff_for_shift(
                self.doctors,
                "day",
                self.config.baseline_doctors_day,
            )
        else:
            active_nurses = self._select_rotating_staff_for_shift(
                self.nurses,
                "night",
                self.config.baseline_nurses_night,
            )
            active_doctors = self._select_rotating_staff_for_shift(
                self.doctors,
                "night",
                self.config.baseline_doctors_night,
            )

        for nurse in active_nurses:
            nurse.is_on_shift = True
            nurse.shift_label = shift_label

        for doctor in active_doctors:
            doctor.is_on_shift = True
            doctor.shift_label = shift_label

    def _sample_discharge_hour(self) -> int:
        hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]
        weights = [0.18, 0.22, 0.22, 0.18, 0.08, 0.05, 0.03, 0.02, 0.02]
        return self.rng.choices(hours, weights=weights, k=1)[0]

    def _sample_admission_hour(self) -> int:
        hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        weights = [0.04, 0.08, 0.12, 0.14, 0.16, 0.16, 0.12, 0.10, 0.05, 0.03]
        return self.rng.choices(hours, weights=weights, k=1)[0]

    def _apply_scheduled_flow_for_current_hour(self):
        day = self.get_current_day()
        hour = self.get_current_hour()
        key = (day, hour)

        census_changed = False

        discharge_ids = self._scheduled_discharges_by_hour.pop(key, [])
        for patient_id in discharge_ids:
            patient = self.agent_index.get(patient_id)
            if isinstance(patient, PatientAgent) and patient.is_active:
                self._discharge_patient(patient)
                census_changed = True

        admission_ids = self._scheduled_admissions_by_hour.pop(key, [])

        hard_upper_census = min(
            self.config.ward_capacity,
            self.config.census_hard_upper,
        )

        for patient_id in admission_ids:
            # Az óránként végrehajtott felvételeknél kemény felső plafont alkalmazunk.
            # Ez engedi a referencia-census körüli természetes ingadozást,
            # de megakadályozza az irreális feltöltődési driftet.
            if self.get_current_patient_count() >= hard_upper_census:
                break

            patient = self.agent_index.get(patient_id)
            if not isinstance(patient, PatientAgent):
                continue
            if patient.is_active:
                continue

            room_id = self._find_first_available_bed()
            if room_id is None:
                break

            self._admit_patient(patient, room_id)
            census_changed = True

        if census_changed:
            self._refresh_assignments_after_census_change()

    def _record_daily_room_colonization_pressure(self, completed_day: int):
        for room_id, patient_ids in self.room_occupants.items():
            active_patients = [
                self.agent_index[pid]
                for pid in patient_ids
                if isinstance(self.agent_index.get(pid), PatientAgent)
                and self.agent_index[pid].is_active
            ]

            susceptible_count = sum(
                1 for p in active_patients
                if p.epi_state == "S"
            )

            infectious_count = sum(
                1 for p in active_patients
                if p.epi_state in {"E_inf", "I_asym", "I_sym"}
            )

            colonized_or_infected_count = sum(
                1 for p in active_patients
                if p.epi_state in {"E_lat", "E_inf", "I_asym", "I_sym", "R"}
            )

            if susceptible_count > 0:
                infectious_pressure = infectious_count / susceptible_count
                colonization_pressure = colonized_or_infected_count / susceptible_count
            else:
                infectious_pressure = None
                colonization_pressure = None

            self.daily_room_pressure_log.append(
                {
                    "run_id": self.config.run_id,
                    "day": int(completed_day),
                    "room_id": room_id,
                    "room_capacity": int(self.room_capacity_map[room_id]),
                    "active_patient_count": int(len(active_patients)),
                    "susceptible_patient_count": int(susceptible_count),
                    "infectious_patient_count": int(infectious_count),
                    "colonized_or_infected_patient_count": int(colonized_or_infected_count),
                    "infectious_pressure": infectious_pressure,
                    "colonization_pressure": colonization_pressure,
                }
            )
    
    def _finalize_previous_day_logs(self, completed_day: int):
        if completed_day < 0:
            return

        admissions_completed_day = self._daily_admissions_counter.get(completed_day, 0)
        discharges_completed_day = self._daily_discharges_counter.get(completed_day, 0)

        census_end_of_day = self.get_current_patient_count()
        occupancy_end_of_day = census_end_of_day / self.config.ward_capacity

        self.daily_census_history.append(census_end_of_day)
        self.daily_flow_log.append(
            {
                "day": int(completed_day),
                "admissions": int(admissions_completed_day),
                "discharges": int(discharges_completed_day),
                "census_end_of_day": int(census_end_of_day),
                "occupancy_end_of_day": float(occupancy_end_of_day),
            }
        )

        self._record_daily_room_colonization_pressure(completed_day)

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
        candidate_rooms = []

        for room_id, capacity in self.room_capacity_map.items():
            occupied = len(self.room_occupants[room_id])
            if occupied < capacity:
                occupancy_ratio = occupied / capacity
                candidate_rooms.append((room_id, occupancy_ratio))

        if not candidate_rooms:
            return None

        min_ratio = min(ratio for _, ratio in candidate_rooms)
        best_rooms = [room_id for room_id, ratio in candidate_rooms if ratio == min_ratio]

        return self.rng.choice(best_rooms)

    def _get_inactive_patients_pool(self) -> list[PatientAgent]:
        return [p for p in self.patients if not p.is_active]

    def _discharge_patient(self, patient: PatientAgent):
        previous_room_id = patient.room_id

        if patient.room_id and patient.unique_id in self.room_occupants[patient.room_id]:
            self.room_occupants[patient.room_id].remove(patient.unique_id)

        patient.is_active = False
        patient.room_id = ""
        patient.admission_day = None
        patient.remaining_los_days = None
        self.total_discharges += 1

        day = self.get_current_day()

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": day,
            "hour": self.get_current_hour(),
            "global_hour": self.get_global_hour(),
            "event": "discharge",
            "patient_id": patient.unique_id,
            "room_id": previous_room_id,
            "census": self.get_current_patient_count(),
            "occupancy": self.get_current_patient_count() / self.config.ward_capacity,
        }
        self._daily_discharges_counter[day] = self._daily_discharges_counter.get(day, 0) + 1
        self._write_flow_event(event)

    def _admit_patient(self, patient: PatientAgent, room_id: str):
        patient.is_active = True
        patient.room_id = room_id
        patient.admission_day = self.get_current_day()
        patient.remaining_los_days = self._sample_los_days(self.config.los_distribution)

        # Az új felvétel új betegként jelenik meg a modellben.
        patient.epi_state = "S"
        patient.is_infectious = False
        patient.is_symptomatic = False
        patient.is_detected = False
        patient.is_isolated = False

        patient.infected_by = None
        patient.infection_tick = None
        patient.latent_until_tick = None
        patient.presymptomatic_until_tick = None
        patient.recovery_tick = None
        patient.death_tick = None

        self.room_occupants[room_id].append(patient.unique_id)
        self.total_admissions += 1

        day = self.get_current_day()

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": day,
            "hour": self.get_current_hour(),
            "global_hour": self.get_global_hour(),
            "event": "admission",
            "patient_id": patient.unique_id,
            "room_id": room_id,
            "census": self.get_current_patient_count(),
            "occupancy": self.get_current_patient_count() / self.config.ward_capacity,
        }
        self._daily_admissions_counter[day] = self._daily_admissions_counter.get(day, 0) + 1
        self._write_flow_event(event)

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

        # Az előző nap lezárása (ha van)
        completed_day = day_idx - 1
        self._finalize_previous_day_logs(completed_day)

        active_patients = [p for p in self.patients if p.is_active]

        # LOS decrement továbbra is napváltáskor történik
        for patient in active_patients:
            if patient.remaining_los_days is not None:
                patient.remaining_los_days -= 1

        # Azon betegek kijelölése, akik ma elbocsáthatók lesznek
        due_for_discharge: list[PatientAgent] = []
        for patient in list(active_patients):
            if patient.remaining_los_days is not None and patient.remaining_los_days <= 0:
                due_for_discharge.append(patient)

        # Nem bocsátjuk el őket azonnal, csak szétosztjuk órákra
        self._scheduled_discharges_by_hour = {
            k: v for k, v in self._scheduled_discharges_by_hour.items() if k[0] != day_idx
        }
        for patient in due_for_discharge:
            discharge_hour = self._sample_discharge_hour()
            key = (day_idx, discharge_hour)
            self._scheduled_discharges_by_hour.setdefault(key, []).append(patient.unique_id)

        # A mai felvételi igényt már a várható discharge-ok után becsüljük
        current_census = self.get_current_patient_count()
        projected_census_after_discharges = current_census - len(due_for_discharge)

        reference_census = int(round(
            self.config.target_bed_occupancy * self.config.ward_capacity
        ))

        soft_lower_census = self.config.census_soft_lower
        soft_upper_census = self.config.census_soft_upper
        hard_upper_census = min(
            self.config.ward_capacity,
            self.config.census_hard_upper,
        )

        # A referencia-census nem kemény napi célérték, hanem hosszabb távú átlagos
        # betegszám. A felvételi igény sávosan működik:
        # - soft_lower alatt aktív visszatöltés történik,
        # - soft_lower és soft_upper között mérsékelt, sztochasztikus felvétel lehetséges,
        # - soft_upper fölött csak discharge-ok csökkentik a censust,
        # - hard_upper fölé nem ütemezünk felvételt.
        if projected_census_after_discharges < soft_lower_census:
            expected_admissions = reference_census - projected_census_after_discharges
        elif projected_census_after_discharges < reference_census:
            expected_admissions = 0.75 * (reference_census - projected_census_after_discharges)
        elif projected_census_after_discharges < soft_upper_census:
            expected_admissions = 0.25 * (soft_upper_census - projected_census_after_discharges)
        else:
            expected_admissions = 0.0

        expected_admissions = max(0.0, expected_admissions)
        daily_admissions = self._sample_poisson(expected_admissions)

        # A napi ütemezés legfeljebb a kemény felső plafonig enged felvételt.
        max_admissions_today = max(0, hard_upper_census - projected_census_after_discharges)

        available_beds_after_discharges = self.config.ward_capacity - projected_census_after_discharges

        daily_admissions = min(
            daily_admissions,
            max_admissions_today,
            available_beds_after_discharges,
        )
        inactive_pool = self._get_inactive_patients_pool()

        self._scheduled_admissions_by_hour = {
            k: v for k, v in self._scheduled_admissions_by_hour.items() if k[0] != day_idx
        }

        selected_patients = inactive_pool[:daily_admissions]
        for patient in selected_patients:
            admission_hour = self._sample_admission_hour()
            key = (day_idx, admission_hour)
            self._scheduled_admissions_by_hour.setdefault(key, []).append(patient.unique_id) 
        
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
        if not self.stream_logs:
            self.visit_events.append(event)
        else:
            self._write_visit_event(event)

        self._update_aggregated_contact_stats(event)

        contact_pair = tuple(sorted([actor_id, target_id]))
        self._recent_contacts[contact_pair] = tick

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
    def _get_seed_candidate_patients(self) -> list[PatientAgent]:
        candidates = []
        for patient in self.patients:
            if self.config.seed_only_active_patients and not patient.is_active:
                continue
            if patient.epi_state != "S":
                continue
            candidates.append(patient)
        return candidates

    def _seed_state_to_flags(self, patient: PatientAgent, seed_state: str):
        patient.is_symptomatic = False
        patient.is_detected = False
        patient.is_isolated = False
        patient.infected_by = "seed"
        patient.infection_tick = self.current_tick
        patient.death_tick = None

        if seed_state == "E_lat":
            patient.epi_state = "E_lat"
            patient.is_infectious = False

            latent_days = sample_gamma_days(
                self.rng,
                self.config.latent_shape,
                self.config.latent_scale_days,
            )
            patient.latent_until_tick = self.current_tick + days_to_ticks(
                latent_days,
                self.config.dt_min,
            )
            patient.presymptomatic_until_tick = None
            patient.recovery_tick = None
            return

        if seed_state == "E_inf":
            patient.epi_state = "E_inf"
            patient.is_infectious = True

            presymptomatic_days = sample_gamma_days(
                self.rng,
                self.config.presymptomatic_shape,
                self.config.presymptomatic_scale_days,
            )
            patient.latent_until_tick = None
            patient.presymptomatic_until_tick = self.current_tick + days_to_ticks(
                presymptomatic_days,
                self.config.dt_min,
            )
            patient.recovery_tick = None
            return

        if seed_state == "I_asym":
            patient.epi_state = "I_asym"
            patient.is_infectious = True

            recovery_days = max(
                4.0,
                sample_gamma_days(
                    self.rng,
                    self.config.recovery_asym_shape,
                    self.config.recovery_asym_scale_days,
                )
            )
            patient.latent_until_tick = None
            patient.presymptomatic_until_tick = None
            patient.recovery_tick = self.current_tick + days_to_ticks(
                recovery_days,
                self.config.dt_min,
            )
            return

        raise ValueError(
            f"Unsupported seed_state='{seed_state}'. "
            f"Allowed: 'E_lat', 'E_inf', 'I_asym'."
        )

    def _force_seed_patient(self, patient: PatientAgent, seed_state: str):
        if not patient.is_active:
            return
        if patient.epi_state != "S":
            return

        self._seed_state_to_flags(patient, seed_state)

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": patient.unique_id,
            "source_id": "seed",
            "source_type": "external",
            "event_type": "forced_seed",
            "new_state": seed_state,
        }
        self.total_infection_events_counter += 1
        self.total_patient_infection_events_counter += 1

        self._write_infection_event(event)
        self.seed_events.append(event)

    def _schedule_initial_seed_introductions(self):
        self._scheduled_seed_introductions = []

        n_seeds = max(0, int(self.config.initial_seed_infections))
        if n_seeds == 0:
            return

        n_days = max(1, int(self.config.seed_in_first_days))
        max_days = max(1, int(self.config.simulation_days))
        n_days = min(n_days, max_days)

        start_hour = int(self.config.seed_start_hour)
        end_hour = int(self.config.seed_end_hour)

        if start_hour < 0 or start_hour > 23 or end_hour < 0 or end_hour > 23:
            raise ValueError("seed_start_hour and seed_end_hour must be between 0 and 23")

        if end_hour < start_hour:
            raise ValueError("seed_end_hour must be >= seed_start_hour")

        for _ in range(n_seeds):
            seed_day = self.rng.randint(0, n_days - 1)
            seed_hour = self.rng.randint(start_hour, end_hour)
            seed_minute = seed_hour * 60
            seed_tick = (seed_day * self.config.ticks_per_day) + (seed_minute // self.config.dt_min)

            self._scheduled_seed_introductions.append(
                {
                    "seed_tick": seed_tick,
                    "seed_state": self.config.seed_state,
                }
            )

        self._scheduled_seed_introductions.sort(key=lambda x: x["seed_tick"])
 
    def _apply_scheduled_seed_introductions(self):
        if not self._scheduled_seed_introductions:
            return

        remaining = []
        for item in self._scheduled_seed_introductions:
            if item["seed_tick"] != self.current_tick:
                remaining.append(item)
                continue

            candidates = self._get_seed_candidate_patients()
            if not candidates:
                continue

            patient = self.rng.choice(candidates)
            self._force_seed_patient(patient, item["seed_state"])

        self._scheduled_seed_introductions = remaining


    def _is_staff_agent(self, agent: BaseHospitalAgent) -> bool:
        return isinstance(agent, (NurseAgent, DoctorAgent))

    def _is_staff_susceptible(self, agent: BaseHospitalAgent) -> bool:
        return self._is_staff_agent(agent) and getattr(agent, "epi_state", None) == "S"

    def _is_staff_infectious(self, agent: BaseHospitalAgent) -> bool:
        return self._is_staff_agent(agent) and getattr(agent, "is_infectious", False) is True

    def _is_patient_susceptible(self, agent: BaseHospitalAgent) -> bool:
        return isinstance(agent, PatientAgent) and agent.is_active and agent.epi_state == "S"

    def _is_patient_infectious(self, agent: BaseHospitalAgent) -> bool:
        return isinstance(agent, PatientAgent) and agent.is_active and agent.epi_state in {"E_inf", "I_asym", "I_sym"}


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

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": patient.unique_id,
            "source_id": source_id,
            "source_type": source_type,
            "event_type": "new_exposure",
            "new_state": "E_lat",
        }
        self.total_infection_events_counter += 1
        self.total_patient_infection_events_counter += 1

        self._write_infection_event(event)

    def _infect_staff_from_contact(
        self,
        staff: BaseHospitalAgent,
        source_id: str,
        source_type: str,
        event_type: str,
    ):
        if not self._is_staff_agent(staff):
            return
        if getattr(staff, "epi_state", None) != "S":
            return

        staff.epi_state = "E_lat"
        staff.is_infectious = False
        staff.is_symptomatic = False
        staff.is_detected = False
        staff.is_isolated = False

        staff.infection_tick = self.current_tick
        staff.infected_by = source_id

        latent_days = sample_gamma_days(
            self.rng,
            self.config.latent_shape,
            self.config.latent_scale_days,
        )
        staff.latent_until_tick = self.current_tick + days_to_ticks(
            latent_days,
            self.config.dt_min,
        )
        staff.presymptomatic_until_tick = None
        staff.recovery_tick = None
        staff.death_tick = None

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": staff.unique_id,
            "source_id": source_id,
            "source_type": source_type,
            "event_type": f"staff_{event_type}",
            "new_state": "E_lat",
        }
        self.total_infection_events_counter += 1
        self.total_staff_infection_events_counter += 1

        if isinstance(staff, NurseAgent):
            self.total_nurse_infection_events_counter += 1
        elif isinstance(staff, DoctorAgent):
            self.total_doctor_infection_events_counter += 1

        self._write_infection_event(event)

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
   
    def _update_staff_infection_state(self, staff: BaseHospitalAgent):
        if not self._is_staff_agent(staff):
            return

        tick = self.current_tick

        if staff.epi_state == "E_lat":
            if staff.latent_until_tick is not None and tick >= staff.latent_until_tick:
                self._progress_staff_e_lat_to_e_inf(staff)
                return

        if staff.epi_state == "E_inf":
            if staff.presymptomatic_until_tick is not None and tick >= staff.presymptomatic_until_tick:
                self._progress_staff_e_inf_to_i_state(staff)
                return

        if staff.epi_state in {"I_asym", "I_sym"}:
            if staff.recovery_tick is not None and tick >= staff.recovery_tick:
                self._process_staff_recovery(staff)
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

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": patient.unique_id,
            "source_id": patient.unique_id,
            "source_type": "self_progression",
            "event_type": "progression",
            "new_state": "E_inf",
        }
        self._write_infection_event(event)

    def _progress_staff_e_lat_to_e_inf(self, staff: BaseHospitalAgent):
        if not self._is_staff_agent(staff):
            return
        if staff.epi_state != "E_lat":
            return

        staff.epi_state = "E_inf"
        staff.is_infectious = True
        staff.is_symptomatic = False
        staff.latent_until_tick = None

        presymptomatic_days = sample_gamma_days(
            self.rng,
            self.config.presymptomatic_shape,
            self.config.presymptomatic_scale_days,
        )
        staff.presymptomatic_until_tick = self.current_tick + days_to_ticks(
            presymptomatic_days,
            self.config.dt_min,
        )

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": staff.unique_id,
            "source_id": staff.unique_id,
            "source_type": "self_progression",
            "event_type": "staff_progression",
            "new_state": "E_inf",
        }
        self._write_infection_event(event)

    def _progress_staff_e_inf_to_i_state(self, staff: BaseHospitalAgent):
        if not self._is_staff_agent(staff):
            return
        if staff.epi_state != "E_inf":
            return

        staff.presymptomatic_until_tick = None

        symptomatic = self.rng.random() < self.config.p_symptomatic

        if symptomatic:
            staff.epi_state = "I_sym"
            staff.is_infectious = True
            staff.is_symptomatic = True

            recovery_days = max(
                3.0,
                sample_gamma_days(
                    self.rng,
                    self.config.recovery_sym_shape,
                    self.config.recovery_sym_scale_days,
                )
            )
            staff.recovery_tick = self.current_tick + days_to_ticks(
                recovery_days,
                self.config.dt_min,
            )

            if self.rng.random() < self.config.staff_symptomatic_stay_home_probability:
                staff.days_off_until_day = self.get_current_day() + self.config.staff_symptomatic_off_duty_days
            else:
                staff.days_off_until_day = None

            new_state = "I_sym"
        else:
            staff.epi_state = "I_asym"
            staff.is_infectious = True
            staff.is_symptomatic = False

            recovery_days = max(
                2.0,
                sample_gamma_days(
                    self.rng,
                    self.config.recovery_asym_shape,
                    self.config.recovery_asym_scale_days,
                )
            )
            staff.recovery_tick = self.current_tick + days_to_ticks(
                recovery_days,
                self.config.dt_min,
            )
            new_state = "I_asym"

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": staff.unique_id,
            "source_id": staff.unique_id,
            "source_type": "self_progression",
            "event_type": "staff_progression",
            "new_state": new_state,
        }
        self._write_infection_event(event)

    def _process_staff_recovery(self, staff: BaseHospitalAgent):
        if not self._is_staff_agent(staff):
            return

        staff.epi_state = "R"
        staff.is_infectious = False
        staff.is_symptomatic = False
        staff.recovery_tick = None
        staff.latent_until_tick = None
        staff.presymptomatic_until_tick = None    
    
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
                3.0,
                sample_gamma_days(
                    self.rng,
                    self.config.recovery_sym_shape,
                    self.config.recovery_sym_scale_days,
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
                2.0,
                sample_gamma_days(
                    self.rng,
                    self.config.recovery_asym_shape,
                    self.config.recovery_asym_scale_days,
                )
            )
            patient.recovery_tick = self.current_tick + days_to_ticks(recovery_days, self.config.dt_min)
            patient.death_tick = None
            new_state = "I_asym"

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": patient.unique_id,
            "source_id": patient.unique_id,
            "source_type": "self_progression",
            "event_type": "progression",
            "new_state": patient.epi_state,
        }
        self._write_infection_event(event)

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

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": patient.unique_id,
            "source_id": patient.unique_id,
            "source_type": "self_progression",
            "event_type": "recovery",
            "new_state": "R",
        }
        self._write_infection_event(event)


    def _process_patient_death(self, patient: PatientAgent):
        if not patient.is_active:
            return

        patient.epi_state = "D"
        patient.is_infectious = False
        patient.is_symptomatic = False
        patient.is_detected = False
        patient.is_isolated = False

        event = {
            "run_id": self.config.run_id,
            "tick": self.current_tick,
            "day": self.get_current_day(),
            "time_min": self.current_time_min,
            "time_str": minute_to_clock_string(self.current_time_min),
            "patient_id": patient.unique_id,
            "source_id": patient.unique_id,
            "source_type": "self_progression",
            "event_type": "death",
            "new_state": "D",
        }
        self.total_patient_death_events_counter += 1

        self._write_infection_event(event)

        self._discharge_patient(patient)

    def _attempt_transmission_from_contact(self, event: dict):
        actor = self.agent_index[event["actor_id"]]
        target = self.agent_index[event["target_id"]]
        duration_min = int(event["duration_min"])
        event_type = event["event_type"]
        
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
                
                if self.rng.random() < p:
                    self._infect_patient_from_contact(
                        actor,
                        source_id=target.unique_id,
                        source_type=target.agent_type,
                        event_type=event_type,
                    )
            return

        # direct staff-patient transmission with mask effect
        if self._is_staff_agent(actor) and isinstance(target, PatientAgent):
            staff = actor
            patient = target

            # infectious patient infects susceptible staff
            if self._is_patient_infectious(patient) and self._is_staff_susceptible(staff):
                infectiousness = self._get_infectiousness_multiplier(patient)
                isolation_multiplier = self.config.isolation_transmission_multiplier if patient.is_isolated else 1.0

                source_multiplier = 1.0
                target_multiplier = 1.0

                if getattr(staff, "mask_compliant", False):
                    target_multiplier = self.config.mask_target_multiplier_hcw

                p_staff = self._get_effective_transmission_prob(
                    self.config.beta_patient_to_hcw_per_5min,
                    duration_min,
                    infectiousness_multiplier=infectiousness,
                    isolation_multiplier=isolation_multiplier * source_multiplier * target_multiplier,
                )

                if self.rng.random() < p_staff:
                    self._infect_staff_from_contact(
                        staff,
                        source_id=patient.unique_id,
                        source_type=patient.agent_type,
                        event_type=event_type,
                    )

            # infectious staff infects susceptible patient
            if self._is_staff_infectious(staff) and self._is_patient_susceptible(patient):
                isolation_multiplier = self.config.isolation_transmission_multiplier if patient.is_isolated else 1.0

                source_multiplier = 1.0
                target_multiplier = 1.0

                if getattr(staff, "mask_compliant", False):
                    source_multiplier = self.config.mask_source_multiplier_hcw

                p_patient = self._get_effective_transmission_prob(
                    self.config.beta_hcw_to_patient_per_5min,
                    duration_min,
                    infectiousness_multiplier=self.config.hcw_relative_infectiousness,
                    isolation_multiplier=isolation_multiplier * source_multiplier * target_multiplier,
                )

                if self.rng.random() < p_patient:
                    self._infect_patient_from_contact(
                        patient,
                        source_id=staff.unique_id,
                        source_type=staff.agent_type,
                        event_type=f"staff_direct_{event_type}",
                    )

        # staff-staff transmission
        if self._is_staff_agent(actor) and self._is_staff_agent(target):

            if self._is_staff_infectious(actor) and self._is_staff_susceptible(target):
                source_multiplier = 1.0
                target_multiplier = 1.0

                if getattr(actor, "mask_compliant", False):
                    source_multiplier = self.config.mask_source_multiplier_hcw
                if getattr(target, "mask_compliant", False):
                    target_multiplier = self.config.mask_target_multiplier_hcw

                p = self._get_effective_transmission_prob(
                    self.config.beta_hcw_hcw_per_5min,
                    duration_min,
                    infectiousness_multiplier=self.config.hcw_relative_infectiousness,
                    isolation_multiplier=source_multiplier * target_multiplier,
                )
                if self.rng.random() < p:
                    self._infect_staff_from_contact(
                        target,
                        source_id=actor.unique_id,
                        source_type=actor.agent_type,
                        event_type="staff_staff",
                    )

            elif self._is_staff_infectious(target) and self._is_staff_susceptible(actor):
                source_multiplier = 1.0
                target_multiplier = 1.0

                if getattr(target, "mask_compliant", False):
                    source_multiplier = self.config.mask_source_multiplier_hcw
                if getattr(actor, "mask_compliant", False):
                    target_multiplier = self.config.mask_target_multiplier_hcw

                p = self._get_effective_transmission_prob(
                    self.config.beta_hcw_hcw_per_5min,
                    duration_min,
                    infectiousness_multiplier=self.config.hcw_relative_infectiousness,
                    isolation_multiplier=source_multiplier * target_multiplier,
                )
                if self.rng.random() < p:
                    self._infect_staff_from_contact(
                        actor,
                        source_id=target.unique_id,
                        source_type=target.agent_type,
                        event_type="staff_staff",
                    )


    # =========================================================
    # Model-level event generation
    # =========================================================
    def _generate_roommate_events(self, tick: int, _time_min: int):
        interval_ticks = max(
            1,
            self.config.roommate_contact_interval_min // self.config.dt_min,
        )

        if tick % interval_ticks != 0:
            return

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
                        duration_min=self.config.roommate_contact_interval_min,
                    )

    def _generate_nurse_station_events(self, tick: int, time_min: int):
        is_random_daytime_tick = (tick % self.config.ticks_per_day) in self._random_nurse_station_ticks
        if not is_random_daytime_tick:
            return

        on_shift_nurses = [n for n in self.nurses if n.is_on_shift]
        if self.rng.random() < 0.3 and len(on_shift_nurses) >= 2:
            n1, n2 = self.rng.sample(on_shift_nurses, k=2)
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

        if time_min % 60 == 0:
            self._update_staff_shift_status()
            self._apply_scheduled_flow_for_current_hour()
            self._refresh_assignments_after_census_change()

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

        ticks_per_hour = 60 // self.config.dt_min
        if (tick + 1) % ticks_per_hour == 0:
            active_cases = sum(
                1 for p in self.patients
                if p.is_active and p.epi_state in {"E_inf", "I_asym", "I_sym"}
            )

            snapshot_event = {
                "run_id": self.config.run_id,
                "tick": tick + 1,
                "day": self.get_current_day(),
                "hour": (((tick + 1) % self.config.ticks_per_day) // ticks_per_hour) % 24,
                "global_hour": (tick + 1) // ticks_per_hour,
                "S": int(sum(1 for p in self.patients if p.is_active and p.epi_state == "S")),
                "E_lat": int(sum(1 for p in self.patients if p.is_active and p.epi_state == "E_lat")),
                "E_inf": int(sum(1 for p in self.patients if p.is_active and p.epi_state == "E_inf")),
                "I_asym": int(sum(1 for p in self.patients if p.is_active and p.epi_state == "I_asym")),
                "I_sym": int(sum(1 for p in self.patients if p.is_active and p.epi_state == "I_sym")),
                "R": int(sum(1 for p in self.patients if p.is_active and p.epi_state == "R")),
                "D": int(sum(1 for p in self.patients if p.epi_state == "D")),
                "active_cases": int(active_cases),
                "census": int(self.get_current_patient_count()),
                "occupancy": float(self.get_current_patient_count() / self.config.ward_capacity),
            }
            self._write_snapshot_event(snapshot_event)

        self.current_tick += 1


# =========================================================
# 6) SIMULATION EXECUTION
# =========================================================

def run_simulation(
    config: SimConfig,
    run_output_dir: str,
    stream_logs: bool = False,
) -> tuple[HospitalContactModel, pd.DataFrame, pd.DataFrame]:
    model = HospitalContactModel(config, run_output_dir=run_output_dir, stream_logs=stream_logs)
    
    total_ticks = config.ticks_per_day * config.simulation_days
    for _ in range(total_ticks):
        model.step()

    model._finalize_previous_day_logs(config.simulation_days - 1)


    if model._aggregated_contact_stats:
        agg_df = pd.DataFrame(list(model._aggregated_contact_stats.values()))
    else:
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

    if not agg_df.empty:
        degree_counter = Counter()
        type_counts = Counter()

        for _, row in agg_df.iterrows():
            degree_counter[row["u_id"]] += 1
            degree_counter[row["v_id"]] += 1

            pair = "".join(sorted([row["u_type"][0].upper(), row["v_type"][0].upper()]))
            type_counts[pair] += int(row["total_contact_count"])

        top5_degree = [
            {"node_id": node_id, "degree": degree}
            for node_id, degree in degree_counter.most_common(5)
        ]
    else:
        type_counts = Counter({"PP": 0, "NP": 0, "DP": 0, "NN": 0, "DN": 0, "DD": 0})
        top5_degree = []

    total_events = int(sum(v["total_contact_count"] for v in model._aggregated_contact_stats.values()))
    unique_edges = int(len(model._aggregated_contact_stats))
    total_infection_events = int(model.total_infection_events_counter)

    summary_df = build_run_summary(
        config=config,
        model=model,
        total_events=total_events,
        unique_edges=unique_edges,
        type_counts=type_counts,
        top5_degree=top5_degree,
        total_infection_events=total_infection_events,
    )

    model._close_stream_writers()
    return model, agg_df, summary_df


def run_single_simulation(
    config: SimConfig,
    run_output_dir: str,
    generate_figures: bool = True,
    stream_logs: bool = False,
) -> dict[str, object]:
    model, agg_df, summary_df = run_simulation(
        config=config,
        run_output_dir=run_output_dir,
        stream_logs=stream_logs,
    )

    visit_log_path = ensure_visit_log_exists(model, run_output_dir)

    csv_paths = export_csvs(
        config=config,
        agg_df=agg_df,
        summary_df=summary_df,
        run_output_dir=run_output_dir,
    )
    infection_path = os.path.join(run_output_dir, "infection_log.csv")
    flow_path = os.path.join(run_output_dir, "flow_log.csv")
    snapshot_path = os.path.join(run_output_dir, "state_snapshot.csv")

    if not stream_logs:
        infection_path = export_infection_csv(model, run_output_dir)
        flow_path = export_flow_csv(model, run_output_dir)
        snapshot_path = export_state_snapshot_csv(model, run_output_dir)

    metadata_path = export_run_metadata(config, run_output_dir)


    analysis_paths = export_analysis_outputs(
        config=config,
        model=model,
        agg_df=agg_df,
        run_output_dir=run_output_dir,
        generate_figures=generate_figures,
    )

    figure_paths: dict[str, str] = {}
    if generate_figures:
        figure_paths = export_figures(config, agg_df, run_output_dir)

    return {
        "summary_df": summary_df,
        "agg_df": agg_df,
        "stream_logs": stream_logs,
        "csv_paths": csv_paths,
        "infection_path": infection_path,
        "flow_path": flow_path,
        "state_snapshot_path": snapshot_path,
        "metadata_path": metadata_path,
        "analysis_paths": analysis_paths,
        "figure_paths": figure_paths,
        "run_output_dir": run_output_dir,
    }


# =========================================================
# 7) DATA AGGREGATION & SUMMARY
# =========================================================

def edge_type(actor_type: str, target_type: str) -> str:
    pair = sorted([actor_type[0].upper(), target_type[0].upper()])
    return "".join(pair)


def build_run_summary(
    config: SimConfig,
    model: HospitalContactModel,
    total_events: int,
    unique_edges: int,
    type_counts: dict[str, int],
    top5_degree: list[dict],
    total_infection_events: int,
) -> pd.DataFrame:
    
    top5_degree_json = json.dumps(top5_degree, ensure_ascii=False)

# =========================
# EPIDEMIC METRICS
# =========================

    states_df = pd.DataFrame(model.state_snapshot_log)

    if not states_df.empty:
        states_df["I_total"] = states_df["I_asym"] + states_df["I_sym"]

        peak_I = states_df["I_total"].max()

        peak_row = states_df.loc[states_df["I_total"].idxmax()]
        time_to_peak_day = peak_row["day"]
        time_to_peak_tick = peak_row["tick"]

        final_row = states_df.iloc[-1]

        final_active_epidemic_burden = (
            final_row["R"]
            + final_row["I_asym"]
            + final_row["I_sym"]
            + final_row["E_lat"]
            + final_row["E_inf"]
        )
    else:
        peak_I = 0
        time_to_peak_day = None
        time_to_peak_tick = None
        final_active_epidemic_burden = 0

    census_history = model.daily_census_history if model.daily_census_history else [model.get_current_patient_count()]
    occupancy_history = [c / config.n_patients for c in census_history]

    all_staff = model.nurses + model.doctors

    final_staff_S = sum(1 for a in all_staff if a.epi_state == "S")
    final_staff_E_lat = sum(1 for a in all_staff if a.epi_state == "E_lat")
    final_staff_E_inf = sum(1 for a in all_staff if a.epi_state == "E_inf")
    final_staff_I_asym = sum(1 for a in all_staff if a.epi_state == "I_asym")
    final_staff_I_sym = sum(1 for a in all_staff if a.epi_state == "I_sym")
    final_staff_R = sum(1 for a in all_staff if a.epi_state == "R")
    final_staff_D = sum(1 for a in all_staff if a.epi_state == "D")

    final_staff_infected_total = (
        final_staff_E_lat
        + final_staff_E_inf
        + final_staff_I_asym
        + final_staff_I_sym
        + final_staff_R
        + final_staff_D
    )

    final_staff_active_infectious = final_staff_I_asym + final_staff_I_sym
    
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
                "room_config_name": config.room_config_name,
                "initial_patient_count": config.initial_patient_count,
                "roommate_contact_interval_min": config.roommate_contact_interval_min,
                "beta_patient_patient_per_5min": config.beta_patient_patient_per_5min,
                "beta_hcw_to_patient_per_5min": config.beta_hcw_to_patient_per_5min,
                "beta_patient_to_hcw_per_5min": config.beta_patient_to_hcw_per_5min,
                "beta_hcw_hcw_per_5min": config.beta_hcw_hcw_per_5min,
                "mask_strategy": config.mask_strategy,
                "targeted_mask_hcw_ids": config.targeted_mask_hcw_ids,
                "initial_seed_infections": config.initial_seed_infections,
                "seed_in_first_days": config.seed_in_first_days,
                "seed_state": config.seed_state,
                "seed_start_hour": config.seed_start_hour,
                "seed_end_hour": config.seed_end_hour,
                "applied_seed_events": int(len(model.seed_events)),
                "first_seed_tick": (
                    int(min(e["tick"] for e in model.seed_events))
                    if model.seed_events else None
                ),
                "last_seed_tick": (
                    int(max(e["tick"] for e in model.seed_events))
                    if model.seed_events else None
                ),
                "total_admissions": model.total_admissions,
                "total_discharges": model.total_discharges,
                "final_patient_count": model.get_current_patient_count(),
                "average_daily_census": float(sum(census_history) / len(census_history)),
                "occupancy_mean_over_run": float(sum(occupancy_history) / len(occupancy_history)),
                "occupancy_min_over_run": float(min(occupancy_history)),
                "occupancy_max_over_run": float(max(occupancy_history)),
                "total_events": int(total_events),
                "unique_edges": int(unique_edges),
                "total_PP_events": int(type_counts.get("PP", 0)),
                "total_PN_events": int(type_counts.get("NP", 0)),
                "total_PD_events": int(type_counts.get("DP", 0)),
                "total_NN_events": int(type_counts.get("NN", 0)),
                "total_ND_events": int(type_counts.get("DN", 0)),
                "total_DD_events": int(type_counts.get("DD", 0)),
                "n_mask_compliant_nurses": int(sum(1 for n in model.nurses if getattr(n, "mask_compliant", False))),
                "n_mask_compliant_doctors": int(sum(1 for d in model.doctors if getattr(d, "mask_compliant", False))),
                "top5_nodes_by_degree": top5_degree_json,
                "final_active_S": int(sum(1 for p in model.patients if p.is_active and p.epi_state == "S")),
                "final_active_E_lat": int(sum(1 for p in model.patients if p.is_active and p.epi_state == "E_lat")),
                "final_active_E_inf": int(sum(1 for p in model.patients if p.is_active and p.epi_state == "E_inf")),
                "final_active_I_asym": int(sum(1 for p in model.patients if p.is_active and p.epi_state == "I_asym")),
                "final_active_I_sym": int(sum(1 for p in model.patients if p.is_active and p.epi_state == "I_sym")),
                "final_active_R": int(sum(1 for p in model.patients if p.is_active and p.epi_state == "R")),

                "cumulative_patient_infections": int(model.total_patient_infection_events_counter),
                "cumulative_staff_infections": int(model.total_staff_infection_events_counter),
                "cumulative_nurse_infections": int(model.total_nurse_infection_events_counter),
                "cumulative_doctor_infections": int(model.total_doctor_infection_events_counter),
                "cumulative_patient_deaths": int(model.total_patient_death_events_counter),
                "total_infection_events": int(total_infection_events),

                # Epidemiological outcome metrics
                # final_epidemic_size is defined as the cumulative number of patient infections
                # during the run, including initially seeded patient infections.
                "final_epidemic_size": int(model.total_patient_infection_events_counter),
                "final_active_epidemic_burden": int(final_active_epidemic_burden),

                "peak_I": int(peak_I),
                "time_to_peak_day": time_to_peak_day,
                "time_to_peak_tick": time_to_peak_tick,
                "final_staff_S": final_staff_S,
                "final_staff_E_lat": final_staff_E_lat,
                "final_staff_E_inf": final_staff_E_inf,
                "final_staff_I_asym": final_staff_I_asym,
                "final_staff_I_sym": final_staff_I_sym,
                "final_staff_R": final_staff_R,
                "final_staff_D": final_staff_D,
                "final_staff_active_infectious": final_staff_active_infectious,
                "final_staff_infected_total": final_staff_infected_total,
            }
        ]
    )
    return summary


# =========================================================
# 8) CSV EXPORT
# =========================================================
def ensure_visit_log_exists(model: HospitalContactModel, run_output_dir: str) -> str:
    visit_path = os.path.join(run_output_dir, "visit_log.csv")

    if os.path.exists(visit_path):
        return visit_path

    visit_df = pd.DataFrame(model.visit_events)
    visit_df.to_csv(visit_path, index=False)
    return visit_path

def export_csvs(
    config: SimConfig,
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

    agg_df.to_csv(agg_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return {
        "visit_log_csv": visit_path,
        "aggregated_edges_csv": agg_path,
        "run_summary_csv": summary_path,
    }

def export_infection_csv(model: HospitalContactModel, run_output_dir: str) -> str:
    infection_path = os.path.join(run_output_dir, "infection_log.csv")
    infection_df = pd.DataFrame(model.infection_events)
    infection_df.to_csv(infection_path, index=False)
    return infection_path

def export_flow_csv(model: HospitalContactModel, run_output_dir: str) -> str:
    flow_path = os.path.join(run_output_dir, "flow_log.csv")
    flow_df = pd.DataFrame(model.flow_log)
    flow_df.to_csv(flow_path, index=False)
    return flow_path


def export_state_snapshot_csv(model: HospitalContactModel, run_output_dir: str) -> str:
    snapshot_path = os.path.join(run_output_dir, "state_snapshot.csv")
    snapshot_df = pd.DataFrame(model.state_snapshot_log)
    snapshot_df.to_csv(snapshot_path, index=False)
    return snapshot_path


def export_run_metadata(config: SimConfig, run_output_dir: str) -> str:
    metadata_path = os.path.join(run_output_dir, "run_metadata.json")

    metadata = {
        "run_id": config.run_id,
        "seed": config.seed,
        "dt_min": config.dt_min,
        "ticks_per_day": config.ticks_per_day,
        "simulation_days": config.simulation_days,
        "room_config_name": config.room_config_name,
        "room_capacity_spec": config.room_capacity_spec,
        "ward_capacity": config.ward_capacity,
        "n_patients_capacity": config.n_patients,
        "n_nurses": config.n_nurses,
        "n_doctors": config.n_doctors,
        "n_rooms": config.n_rooms,
        "initial_patient_count": config.initial_patient_count,
        "target_bed_occupancy": config.target_bed_occupancy,
        "mean_los_days": config.mean_los_days,
        "census_soft_lower": config.census_soft_lower,
        "census_soft_upper": config.census_soft_upper,
        "census_hard_upper": config.census_hard_upper,
        "los_distribution": config.los_distribution,
        "initial_remaining_los_distribution": config.initial_remaining_los_distribution,
        "los_gamma_shape": config.los_gamma_shape,
        "los_min_days": config.los_min_days,
        "los_max_days": config.los_max_days,
        "initial_seed_infections": config.initial_seed_infections,
        "seed_in_first_days": config.seed_in_first_days,
        "seed_state": config.seed_state,
        "seed_start_hour": config.seed_start_hour,
        "seed_end_hour": config.seed_end_hour,
        "seed_only_active_patients": config.seed_only_active_patients,
        "p_symptomatic": config.p_symptomatic,
        "infection_fatality_ratio": config.infection_fatality_ratio,
        "latent_shape": config.latent_shape,
        "latent_scale_days": config.latent_scale_days,
        "presymptomatic_shape": config.presymptomatic_shape,
        "presymptomatic_scale_days": config.presymptomatic_scale_days,
        "recovery_asym_shape": config.recovery_asym_shape,
        "recovery_asym_scale_days": config.recovery_asym_scale_days,
        "recovery_sym_shape": config.recovery_sym_shape,
        "recovery_sym_scale_days": config.recovery_sym_scale_days,
        "death_shape": config.death_shape,
        "death_scale_days": config.death_scale_days,
        "beta_patient_patient_per_5min": config.beta_patient_patient_per_5min,
        "beta_hcw_to_patient_per_5min": config.beta_hcw_to_patient_per_5min,
        "beta_patient_to_hcw_per_5min": config.beta_patient_to_hcw_per_5min,
        "e_inf_relative_infectiousness": config.e_inf_relative_infectiousness,
        "i_asym_relative_infectiousness": config.i_asym_relative_infectiousness,
        "i_sym_relative_infectiousness": config.i_sym_relative_infectiousness,
        "isolation_transmission_multiplier": config.isolation_transmission_multiplier,
        "mask_strategy": config.mask_strategy,
        "targeted_mask_hcw_ids": config.targeted_mask_hcw_ids,
        "mask_compliance_hcw": config.mask_compliance_hcw,
        "mask_source_multiplier_hcw": config.mask_source_multiplier_hcw,
        "mask_target_multiplier_hcw": config.mask_target_multiplier_hcw,
        "beta_hcw_hcw_per_5min": config.beta_hcw_hcw_per_5min,
        "hcw_relative_infectiousness": config.hcw_relative_infectiousness,
        "staff_symptomatic_stay_home_probability": config.staff_symptomatic_stay_home_probability,
        "staff_symptomatic_off_duty_days": config.staff_symptomatic_off_duty_days,
        "roommate_contact_interval_min": config.roommate_contact_interval_min,
        }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path

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


def plot_timeseries(config: SimConfig, visit_log_path: str, out_path: str):
    visit_df = load_visit_log_df(visit_log_path)

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


def export_figures(config: SimConfig, agg_df: pd.DataFrame, run_output_dir: str | None = None):
    if run_output_dir is None:
        run_output_dir = config.output_dir

    fig_dir = os.path.join(run_output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    network_path = os.path.join(fig_dir, "network.png")
    timeseries_path = os.path.join(fig_dir, "timeseries.png")
    degree_hist_path = os.path.join(fig_dir, "degree_hist.png")

    plot_network(config, agg_df, network_path)
    visit_log_path = os.path.join(run_output_dir, "visit_log.csv")
    plot_timeseries(config, visit_log_path, timeseries_path)
    plot_degree_hist(config, agg_df, degree_hist_path)

    return {
        "network_png": network_path,
        "timeseries_png": timeseries_path,
        "degree_hist_png": degree_hist_path,
    }

# =========================================================
# 9/B) OUTPUT ANALYSIS
# =========================================================
def load_visit_log_df(visit_log_path: str) -> pd.DataFrame:
    if not os.path.exists(visit_log_path):
        return pd.DataFrame(
            columns=[
                "run_id",
                "tick",
                "day",
                "time_min",
                "time_str",
                "actor_id",
                "actor_type",
                "target_id",
                "target_type",
                "room_id",
                "event_type",
                "duration_min",
            ]
        )

    try:
        df = pd.read_csv(visit_log_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=[
                "run_id",
                "tick",
                "day",
                "time_min",
                "time_str",
                "actor_id",
                "actor_type",
                "target_id",
                "target_type",
                "room_id",
                "event_type",
                "duration_min",
            ]
        )

    return df

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


def build_daily_timeseries_dataset(config: SimConfig, visit_log_path: str) -> pd.DataFrame:
    df = load_visit_log_df(visit_log_path)

    if df.empty:
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


def build_role_pair_tables(visit_log_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_visit_log_df(visit_log_path)

    if df.empty:
        summary = pd.DataFrame(columns=["role_pair", "count", "ratio"])
        matrix_counts = pd.DataFrame(
            0,
            index=["patient", "nurse", "doctor"],
            columns=["patient", "nurse", "doctor"],
        )
        matrix_ratios = matrix_counts.astype(float)
        return summary, matrix_counts, matrix_ratios
    
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

def build_staff_patient_contact_table(visit_log_path: str) -> pd.DataFrame:
    df = load_visit_log_df(visit_log_path)

    columns = [
        "staff_id",
        "staff_type",
        "unique_patient_contacts",
        "total_patient_contact_events",
        "is_high_mobility_hcw",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    staff_patient_rows = df[
        (
            df["actor_type"].isin(["nurse", "doctor"])
            & (df["target_type"] == "patient")
        )
        | (
            (df["actor_type"] == "patient")
            & df["target_type"].isin(["nurse", "doctor"])
        )
    ].copy()

    if staff_patient_rows.empty:
        return pd.DataFrame(columns=columns)

    def _staff_id(row):
        if row["actor_type"] in {"nurse", "doctor"}:
            return row["actor_id"]
        return row["target_id"]

    def _staff_type(row):
        if row["actor_type"] in {"nurse", "doctor"}:
            return row["actor_type"]
        return row["target_type"]

    def _patient_id(row):
        if row["actor_type"] == "patient":
            return row["actor_id"]
        return row["target_id"]

    staff_patient_rows["staff_id"] = staff_patient_rows.apply(_staff_id, axis=1)
    staff_patient_rows["staff_type"] = staff_patient_rows.apply(_staff_type, axis=1)
    staff_patient_rows["patient_id"] = staff_patient_rows.apply(_patient_id, axis=1)

    result = (
        staff_patient_rows
        .groupby(["staff_id", "staff_type"], as_index=False)
        .agg(
            unique_patient_contacts=("patient_id", "nunique"),
            total_patient_contact_events=("patient_id", "size"),
        )
        .sort_values(
            ["unique_patient_contacts", "total_patient_contact_events"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    mean_unique_patient_contacts = result["unique_patient_contacts"].mean()
    result["is_high_mobility_hcw"] = (
        result["unique_patient_contacts"] > mean_unique_patient_contacts
    )

    return result[columns]


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

def build_daily_room_pressure_dataframe(model: HospitalContactModel) -> pd.DataFrame:
    columns = [
        "run_id",
        "day",
        "room_id",
        "room_capacity",
        "active_patient_count",
        "susceptible_patient_count",
        "infectious_patient_count",
        "colonized_or_infected_patient_count",
        "infectious_pressure",
        "colonization_pressure",
    ]

    if not model.daily_room_pressure_log:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(model.daily_room_pressure_log)[columns]

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
    agg_df: pd.DataFrame,
    run_output_dir: str,
    generate_figures: bool = True,
):
    visit_log_path = os.path.join(run_output_dir, "visit_log.csv")

    ts_df = build_daily_timeseries_dataset(config, visit_log_path)
    role_pair_summary_df, role_pair_matrix_counts_df, role_pair_matrix_ratios_df = build_role_pair_tables(visit_log_path)
    degree_node_df, degree_role_df = build_degree_tables(agg_df)
    staff_patient_contacts_df = build_staff_patient_contact_table(visit_log_path)
    daily_flow_df = build_daily_flow_dataframe(model)
    daily_room_pressure_df = build_daily_room_pressure_dataframe(model)
    
    ts_path = os.path.join(run_output_dir, "timeseries_daily.csv")
    role_pair_summary_path = os.path.join(run_output_dir, "role_pair_summary.csv")
    role_pair_counts_path = os.path.join(run_output_dir, "role_pair_matrix_counts.csv")
    role_pair_ratios_path = os.path.join(run_output_dir, "role_pair_matrix_ratios.csv")
    degree_node_path = os.path.join(run_output_dir, "degree_summary_by_node.csv")
    degree_role_path = os.path.join(run_output_dir, "degree_summary_by_role.csv")
    staff_patient_contacts_path = os.path.join(run_output_dir, "staff_patient_contacts.csv")
    daily_flow_path = os.path.join(run_output_dir, "daily_flow.csv")
    daily_room_pressure_path = os.path.join(run_output_dir, "daily_room_pressure.csv")

    ts_df.to_csv(ts_path, index=False)
    role_pair_summary_df.to_csv(role_pair_summary_path, index=False)
    role_pair_matrix_counts_df.to_csv(role_pair_counts_path, index=True)
    role_pair_matrix_ratios_df.to_csv(role_pair_ratios_path, index=True)
    degree_node_df.to_csv(degree_node_path, index=False)
    degree_role_df.to_csv(degree_role_path, index=False)
    staff_patient_contacts_df.to_csv(staff_patient_contacts_path, index=False)
    daily_flow_df.to_csv(daily_flow_path, index=False)
    daily_room_pressure_df.to_csv(daily_room_pressure_path, index=False)

    result = {
        "timeseries_daily_csv": ts_path,
        "role_pair_summary_csv": role_pair_summary_path,
        "role_pair_matrix_counts_csv": role_pair_counts_path,
        "role_pair_matrix_ratios_csv": role_pair_ratios_path,
        "degree_summary_by_node_csv": degree_node_path,
        "degree_summary_by_role_csv": degree_role_path,
        "staff_patient_contacts_csv": staff_patient_contacts_path,
        "daily_flow_csv": daily_flow_path,
        "daily_room_pressure_csv": daily_room_pressure_path,
    }

    if generate_figures:
        fig_dir = os.path.join(run_output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        daily_events_fig_path = os.path.join(fig_dir, "daily_events.png")
        daily_unique_patients_fig_path = os.path.join(fig_dir, "daily_unique_patients.png")
        role_pair_bar_fig_path = os.path.join(fig_dir, "role_pair_bar.png")
        edge_weight_hist_fig_path = os.path.join(fig_dir, "edge_weight_hist.png")
        daily_flow_fig_path = os.path.join(fig_dir, "daily_flow.png")

        plot_daily_events_analysis(ts_df, daily_events_fig_path)
        plot_daily_unique_patients(ts_df, daily_unique_patients_fig_path)
        plot_role_pair_bar(role_pair_summary_df, role_pair_bar_fig_path)
        plot_edge_weight_histogram(agg_df, edge_weight_hist_fig_path)
        plot_daily_flow(daily_flow_df, daily_flow_fig_path)

        result.update(
            {
                "daily_events_png": daily_events_fig_path,
                "daily_unique_patients_png": daily_unique_patients_fig_path,
                "role_pair_bar_png": role_pair_bar_fig_path,
                "edge_weight_hist_png": edge_weight_hist_fig_path,
                "daily_flow_png": daily_flow_fig_path,
            }
        )

    return result

# =========================================================
# 10) CLI & MAIN
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mesa hospital contact-network prototype with infection dynamics")
    parser.add_argument("--seed", type=int, default=None, help="Deprecated alias for --base_seed")
    parser.add_argument("--base_seed", type=int, default=DEFAULT_SEED, help="Base seed for reproducible runs")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs to execute")
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Run identifier (default: UNIX timestamp)",
    )
    parser.add_argument("--initial-seed-infections", type=int, default=1)
    parser.add_argument("--seed-in-first-days", type=int, default=1)
    parser.add_argument("--seed-state", type=str, default="I_asym", choices=["E_lat", "E_inf", "I_asym"])
    parser.add_argument("--seed-start-hour", type=int, default=9)
    parser.add_argument("--seed-end-hour", type=int, default=9)

    parser.add_argument("--mask-compliance-hcw", type=float, default=0.20)
    parser.add_argument(
        "--mask-strategy",
        type=str,
        default="random",
        choices=["random", "targeted_ids"],
    )
    parser.add_argument(
        "--beta-patient-patient-per-5min",
        type=float,
        default=SimConfig.beta_patient_patient_per_5min,
        help="Transmission probability per 5-minute patient-patient contact.",
    )
    parser.add_argument(
        "--beta-hcw-to-patient-per-5min",
        type=float,
        default=SimConfig.beta_hcw_to_patient_per_5min,
        help="Transmission probability per 5-minute HCW-to-patient contact.",
    )
    parser.add_argument(
        "--beta-patient-to-hcw-per-5min",
        type=float,
        default=SimConfig.beta_patient_to_hcw_per_5min,
        help="Transmission probability per 5-minute patient-to-HCW contact.",
    )
    parser.add_argument(
        "--beta-hcw-hcw-per-5min",
        type=float,
        default=SimConfig.beta_hcw_hcw_per_5min,
        help="Transmission probability per 5-minute HCW-HCW contact.",
    )
    parser.add_argument(
        "--targeted-mask-hcw-ids",
        type=str,
        default="",
        help="Comma-separated staff IDs for targeted mask intervention, e.g. nurse_1,nurse_2,doctor_0",
    )
    parser.add_argument(
        "--room-config",
        type=str,
        default="baseline",
        choices=list(ROOM_CONFIGS.keys()),
    )

    parser.add_argument("--mean-los-days", type=float, default=7.69)
    parser.add_argument(
        "--los-distribution",
        type=str,
        default="gamma",
        choices=["fixed", "gamma", "exponential"],
    )
    parser.add_argument(
        "--initial-remaining-los-distribution",
        type=str,
        default="random_point_in_stay",
        choices=["random_point_in_stay", "discrete_uniform_1_8", "fixed", "gamma", "exponential"],
    )
    parser.add_argument("--los-gamma-shape", type=float, default=2.0)
    parser.add_argument("--los-min-days", type=int, default=1)
    parser.add_argument("--los-max-days", type=int, default=30)
    parser.add_argument("--roommate-contact-interval-min", type=int, default=120)
    return parser.parse_args()


def resolve_base_seed(args: argparse.Namespace) -> int:
    if args.seed is not None:
        return args.seed
    return args.base_seed


def export_all_runs_summary(summary_frames: list[pd.DataFrame], batch_output_dir: str) -> tuple[str, pd.DataFrame]:
    if summary_frames:
        all_runs_summary_df = pd.concat(summary_frames, ignore_index=True)
    else:
        all_runs_summary_df = pd.DataFrame(columns=["run_id", "seed"])

    output_path = os.path.join(batch_output_dir, "all_runs_summary.csv")
    all_runs_summary_df.to_csv(output_path, index=False)
    return output_path, all_runs_summary_df


def print_single_run_report(result: dict[str, object]):
    summary_df = result["summary_df"]
    csv_paths = result["csv_paths"]
    analysis_paths = result["analysis_paths"]
    figure_paths = result["figure_paths"]
    infection_path = result["infection_path"]
    flow_path = result["flow_path"]
    state_snapshot_path = result["state_snapshot_path"]
    metadata_path = result["metadata_path"]
    run_output_dir = result["run_output_dir"]

    total_events = int(summary_df.loc[0, "total_events"])
    unique_edges = int(summary_df.loc[0, "unique_edges"])

    print("\n=== Simulation finished ===")
    print(f"run_id={summary_df.loc[0, 'run_id']} | seed={summary_df.loc[0, 'seed']}")
    print(f"total_events={total_events}, unique_edges={unique_edges}")
    print(f"\nRun-specific output directory: {run_output_dir}")
    print("\nOutput files:")

    print(f"- {csv_paths['visit_log_csv']}")
    
    print(f"- {csv_paths['aggregated_edges_csv']}")
    print(f"- {csv_paths['run_summary_csv']}")
    print(f"- {infection_path}")
    print(f"- {flow_path}")
    print(f"- {state_snapshot_path}")
    print(f"- {metadata_path}")
    print(f"- {analysis_paths['daily_flow_csv']}")
    print(f"- {analysis_paths['daily_room_pressure_csv']}")
    print(f"- {analysis_paths['degree_summary_by_node_csv']}")
    print(f"- {analysis_paths['degree_summary_by_role_csv']}")
    print(f"- {analysis_paths['role_pair_matrix_counts_csv']}")
    print(f"- {analysis_paths['staff_patient_contacts_csv']}")
    print(f"- {analysis_paths['role_pair_matrix_ratios_csv']}")
    print(f"- {analysis_paths['role_pair_summary_csv']}")
    print(f"- {analysis_paths['timeseries_daily_csv']}")

    if figure_paths:
        print("\nFigure files:")
        print(f"- {figure_paths['network_png']}")
        print(f"- {figure_paths['timeseries_png']}")
        print(f"- {figure_paths['degree_hist_png']}")
        print(f"- {analysis_paths['daily_events_png']}")
        print(f"- {analysis_paths['daily_unique_patients_png']}")
        print(f"- {analysis_paths['role_pair_bar_png']}")
        print(f"- {analysis_paths['edge_weight_hist_png']}")
        print(f"- {analysis_paths['daily_flow_png']}")


def main():
    args = parse_args()
    if args.n_runs < 1:
        raise ValueError("--n_runs must be at least 1")

    base_seed = resolve_base_seed(args)

    if args.n_runs == 1:
        run_id = args.run_id or str(int(datetime.utcnow().timestamp()))
        config = SimConfig(
            seed=base_seed,
            run_id=run_id,
            room_config_name=args.room_config,
            initial_seed_infections=args.initial_seed_infections,
            seed_in_first_days=args.seed_in_first_days,
            seed_state=args.seed_state,
            seed_start_hour=args.seed_start_hour,
            seed_end_hour=args.seed_end_hour,
            mask_strategy=args.mask_strategy,
            targeted_mask_hcw_ids=args.targeted_mask_hcw_ids,
            mask_compliance_hcw=args.mask_compliance_hcw,
            mean_los_days=args.mean_los_days,
            los_distribution=args.los_distribution,
            initial_remaining_los_distribution=args.initial_remaining_los_distribution,
            los_gamma_shape=args.los_gamma_shape,
            los_min_days=args.los_min_days,
            los_max_days=args.los_max_days,
            roommate_contact_interval_min=args.roommate_contact_interval_min,
            beta_patient_patient_per_5min=args.beta_patient_patient_per_5min,
            beta_hcw_to_patient_per_5min=args.beta_hcw_to_patient_per_5min,
            beta_patient_to_hcw_per_5min=args.beta_patient_to_hcw_per_5min,
            beta_hcw_hcw_per_5min=args.beta_hcw_hcw_per_5min,
        )
        
        run_output_dir = ensure_unique_output_dir(
            build_run_output_dir(config.output_dir, config.run_id, config.seed)
        )
        print("RUN_OUTPUT_DIR:", run_output_dir)
        print("RUN CONFIG:", {
            "room_config_name": config.room_config_name,
            "ward_capacity": config.ward_capacity,
            "n_rooms": config.n_rooms,
            "mask_strategy": config.mask_strategy,
            "targeted_mask_hcw_ids": config.targeted_mask_hcw_ids,
            "initial_seed_infections": config.initial_seed_infections,
            "seed_in_first_days": config.seed_in_first_days,
            "seed_state": config.seed_state,
            "seed_start_hour": config.seed_start_hour,
            "seed_end_hour": config.seed_end_hour,
            "mask_compliance_hcw": config.mask_compliance_hcw,
            "mean_los_days": config.mean_los_days,
            "los_distribution": config.los_distribution,
            "initial_remaining_los_distribution": config.initial_remaining_los_distribution,
            "los_gamma_shape": config.los_gamma_shape,
            "los_min_days": config.los_min_days,
            "los_max_days": config.los_max_days,
            "roommate_contact_interval_min": config.roommate_contact_interval_min,
            "beta_patient_patient_per_5min": config.beta_patient_patient_per_5min,
            "beta_hcw_to_patient_per_5min": config.beta_hcw_to_patient_per_5min,
            "beta_patient_to_hcw_per_5min": config.beta_patient_to_hcw_per_5min,
            "beta_hcw_hcw_per_5min": config.beta_hcw_hcw_per_5min,
        })

        result = run_single_simulation(
            config=config,
            run_output_dir=run_output_dir,
            generate_figures=True,
            stream_logs=False,
        )
        print_single_run_report(result)
        return

    batch_summary_frames: list[pd.DataFrame] = []

    batch_output_dir = ensure_unique_output_dir(
        build_batch_output_dir(SimConfig().output_dir, base_seed, args.n_runs)
    )

    for run_index in range(args.n_runs):
        run_number = run_index + 1
        seed = base_seed + run_index
        run_id = f"run_{run_number:03d}"
        config = SimConfig(
            seed=seed,
            run_id=run_id,
            room_config_name=args.room_config,
            initial_seed_infections=args.initial_seed_infections,
            seed_in_first_days=args.seed_in_first_days,
            seed_state=args.seed_state,
            seed_start_hour=args.seed_start_hour,
            seed_end_hour=args.seed_end_hour,
            mask_strategy=args.mask_strategy,
            targeted_mask_hcw_ids=args.targeted_mask_hcw_ids,
            mask_compliance_hcw=args.mask_compliance_hcw,
            mean_los_days=args.mean_los_days,
            los_distribution=args.los_distribution,
            initial_remaining_los_distribution=args.initial_remaining_los_distribution,
            los_gamma_shape=args.los_gamma_shape,
            los_min_days=args.los_min_days,
            los_max_days=args.los_max_days,
            roommate_contact_interval_min=args.roommate_contact_interval_min,
            beta_patient_patient_per_5min=args.beta_patient_patient_per_5min,
            beta_hcw_to_patient_per_5min=args.beta_hcw_to_patient_per_5min,
            beta_patient_to_hcw_per_5min=args.beta_patient_to_hcw_per_5min,
            beta_hcw_hcw_per_5min=args.beta_hcw_hcw_per_5min,
        )
        run_output_dir = ensure_unique_output_dir(
            build_batch_run_output_dir(batch_output_dir, run_number)
        )

        print(
            f"Starting run {run_number}/{args.n_runs} with seed {seed} | "
            f"beta_PP={config.beta_patient_patient_per_5min}, "
            f"beta_HCW_to_P={config.beta_hcw_to_patient_per_5min}, "
            f"beta_P_to_HCW={config.beta_patient_to_hcw_per_5min}, "
            f"beta_HCW_HCW={config.beta_hcw_hcw_per_5min}"
        )

        try:
            result = run_single_simulation(
                config=config,
                run_output_dir=run_output_dir,
                generate_figures=False,
                stream_logs=True,
            )
            batch_summary_frames.append(result["summary_df"])
            print(f"Finished run {run_number}/{args.n_runs}")
        except Exception as exc:
            print(f"Run {run_number}/{args.n_runs} failed with seed {seed}: {exc}")
            continue

    print(f"batch_output_dir={batch_output_dir}")
    
    all_runs_summary_path, all_runs_summary_df = export_all_runs_summary(batch_summary_frames, batch_output_dir)
    
    print("\n=== Batch execution finished ===")
    print(f"successful_runs={len(all_runs_summary_df)}, requested_runs={args.n_runs}")
    print(f"all_runs_summary={all_runs_summary_path}")
    
    
if __name__ == "__main__":
    main()
