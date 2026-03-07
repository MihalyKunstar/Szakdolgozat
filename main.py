#!/usr/bin/env python3
"""
Single-file Mesa prototype: 1 ward hospital contact network generation (NO infection dynamics).

REFACTORED VERSION: More agent-centric behavior with explicit agent-level state machines.
Nurses and doctors now have their own step() methods and decide their behavior based on
time blocks and assignments. The model coordinates the environment and event logging,
but agents drive most contact generation.

Outputs:
- outputs/visit_log.csv
- outputs/aggregated_edges.csv
- outputs/run_summary.csv
- outputs/figures/network.png
- outputs/figures/timeseries.png
- outputs/figures/degree_hist.png
"""

# =========================================================
# 1) IMPORTS & CONSTANTS: Libraries and simulation parameters
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


# Simulation parameters
DEFAULT_SEED = 42
DEFAULT_DT_MIN = 5  # Time interval per tick in minutes
MINUTES_PER_DAY = 24 * 60
TICKS_PER_DAY = MINUTES_PER_DAY // DEFAULT_DT_MIN  # Total ticks in a 24-hour day (288)

N_PATIENTS = 50
N_NURSES = 10
N_DOCTORS = 5
N_ROOMS = 13
DURATION_MIN_DEFAULT = 5  # Default duration of any contact event

# Room capacity specification (beds per room):
# room_0, room_1 => 2 beds each
# room_2 to room_11 => 4 beds each
# room_12 => 6 beds
# Total: 2*2 + 10*4 + 6 = 50 beds (matching N_PATIENTS)
ROOM_CAPACITY_SPEC = [2, 2] + [4] * 10 + [6]

# Scheduled care activity time blocks [start_minute, end_minute), local daytime:
DOCTOR_BLOCKS = [(7 * 60, 8 * 60), (15 * 60, 16 * 60)]  # 7-8 AM, 3-4 PM
NURSE_ROUNDS_BLOCKS = [(6 * 60, 7 * 60), (16 * 60, 17 * 60)]  # 6-7 AM, 4-5 PM
FEEDING_BLOCKS = [(8 * 60, 9 * 60), (12 * 60, 13 * 60), (18 * 60, 19 * 60)]  # 8-9 AM, 12-1 PM, 6-7 PM
AD_HOC_BLOCK = (9 * 60, 15 * 60)  # Daytime window for random interactions (9 AM - 3 PM)
HANDOVER_BLOCKS = [(6 * 60 + 55, 7 * 60 + 5), (15 * 60 + 55, 16 * 60 + 5)]  # Shift handover windows


# =========================================================
# 2) PARAMETERS & CONFIGURATION: SimConfig dataclass
# =========================================================
@dataclass
class SimConfig:
    """
    Simulation configuration container.
    
    Attributes:
        seed: Random seed for reproducibility
        run_id: Unique identifier for this simulation run
        dt_min: Time interval in minutes for each tick
        ticks_per_day: Number of ticks in a 24-hour cycle
        n_patients, n_nurses, n_doctors, n_rooms: Agent and room counts
        feeding_coverage_min/max: Range for patient feeding coverage
        p_ad_hoc_tick: Probability of ad-hoc contact event per tick
        p_roommate_event_per_room_per_hour: Probability of roommate interaction
        output_dir: Directory for output files
    """
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
# 3) TIME UTILITIES: Tick-to-time conversions and time block checks
# =========================================================
def build_run_output_dir(base_output_dir: str, run_id: str, seed: int) -> str:
    """Build a unique run-specific output directory path.
    
    Creates a directory name that includes a human-readable timestamp, seed, and run_id
    to ensure each simulation run has its own isolated output folder and previous
    results are never overwritten. This allows running multiple simulations without
    losing any results to accidental overwrites.
    
    Folder naming convention: YYYYMMDD_HHMMSS_seedN_<run_id>
    Example: outputs/20260307_141530_seed42_my_experiment
    
    Args:
        base_output_dir: Base output directory (e.g., "outputs")
        run_id: Unique identifier for this run (user-provided or timestamp-based)
        seed: Random seed value for reproducibility
    
    Returns:
        Path to run-specific directory that uniquely identifies this simulation run.
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp_str}_seed{seed}_{run_id}"
    return os.path.join(base_output_dir, run_dir_name)


def tick_to_time_min(tick: int, dt_min: int) -> int:
    """Convert simulation tick number to minutes since start of day."""
    return tick * dt_min


def minutes_to_hhmm(total_minutes: int) -> str:
    """Convert minutes since start of day to HH:MM format."""
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def in_any_block(time_min: int, blocks: list[tuple[int, int]]) -> bool:
    """Check if time_min falls within any of the given time blocks."""
    return any(start <= time_min < end for start, end in blocks)


def in_block(time_min: int, block: tuple[int, int]) -> bool:
    """Check if time_min falls within the given time block [start, end)."""
    start, end = block
    return start <= time_min < end


def block_index(time_min: int, blocks: list[tuple[int, int]]) -> int | None:
    """Return the index of the block containing time_min, or None if outside all blocks."""
    for idx, (start, end) in enumerate(blocks):
        if start <= time_min < end:
            return idx
    return None


# =========================================================
# 4) AGENT CLASSES: Patient, Nurse, Doctor, Base
# =========================================================
class BaseHospitalAgent(Agent):
    """Base class for all hospital agents (patients, nurses, doctors)."""
    def __init__(self, model: Model, unique_id: str, agent_type: str):
        super().__init__(model)
        self.unique_id = unique_id
        self.agent_type = agent_type


class PatientAgent(BaseHospitalAgent):
    """
    Patient agent. Assigned to a room and involved in care contact events.
    
    Patients remain mostly passive. They may participate in roommate interactions,
    but these are still model-managed for simplicity.
    """
    def __init__(self, model: Model, unique_id: str, room_id: str):
        super().__init__(model, unique_id, "patient")
        self.room_id = room_id
    
    def step(self):
        """Patient step: currently passive. Roommate interactions are model-managed."""
        pass


class NurseAgent(BaseHospitalAgent):
    """
    Nurse agent with explicit state machine and own step() behavior.
    
    State machine:
    - "rounding": Visiting assigned patients during nurse round time blocks.
    - "feeding": Serving designated patients during feeding blocks (if feeder).
    - "handover": At nurse station during shift handover times.
    - "ad_hoc": Occasional unscheduled patient visits during daytime.
    - "station": At nurse station or doing administrative work.
    - "idle": Off-duty or between assignments.
    
    IMPROVEMENTS: Agents now maintain persistent block-level visit plans that are
    precomputed at each block start and distributed across ticks with smoothing.
    Visits are pre-assigned to specific ticks with jitter to eliminate quota-based
    spikiness and create more realistic temporal distributions.
    
    Attributes:
        caseload_rooms: List of room IDs this nurse is responsible for.
        is_active_feeder: Whether this nurse is selected as a feeder for the day.
        current_state: Current state (one of the above).
        
        Block planning (persistent per block):
        round_visits_per_tick: Dict[int, List[str]] mapping tick to patients to visit.
        round_block_started_at_tick: Tick when current round block plan was created.
        
        feeding_visits_per_tick: Dict[int, List[str]] mapping tick to patients to feed.
        feeding_block_started_at_tick: Tick when current feeding block plan was created.
        
        handover_block_idx: Index of the current handover block (-1 if not in one).
        handover_initiated_this_block: Whether initiated contact in current block.
    """
    def __init__(self, model: Model, unique_id: str):
        super().__init__(model, unique_id, "nurse")
        self.caseload_rooms: list[str] = []  # List of room IDs this nurse is responsible for
        self.is_active_feeder: bool = False  # Whether this nurse participates in feeding
        
        # State machine
        self.current_state: str = "idle"
        
        # Block-level persistent visit plans (tick -> list of patients)
        self.round_visits_per_tick: dict[int, list[str]] = {}  # Pre-assigned visits per tick
        self.round_block_started_at_tick: int = -1  # When current plan was created
        
        self.feeding_visits_per_tick: dict[int, list[str]] = {}  # Pre-assigned feedings per tick
        self.feeding_block_started_at_tick: int = -1  # When current plan was created
        
        # Handover state (track block to reset properly)
        self.handover_block_idx: int = -1  # Current handover block index
        self.handover_initiated_this_block: bool = False
    
    def step(self):
        """Execute one tick of nurse behavior based on current time and state."""
        time_min = self.model.get_current_time_min()
        tick = self.model.current_tick
        
        # Determine what state we should be in based on current time
        self.current_state = self._get_current_state(time_min)
        
        # Execute behavior appropriate to current state
        if self.current_state == "rounding":
            self._handle_rounding(tick, time_min)
        elif self.current_state == "feeding":
            self._handle_feeding(tick, time_min)
        elif self.current_state == "handover":
            self._handle_handover(tick, time_min)
        elif self.current_state == "ad_hoc":
            self._handle_ad_hoc(tick, time_min)
        # station and idle states don't generate proactive contacts
    
    def _get_current_state(self, time_min: int) -> str:
        """Determine the nurse's current state based on time of day.
        
        Priority order: handover > feeding > rounding > ad_hoc > station/idle
        """
        # Check handover first
        if in_any_block(time_min, HANDOVER_BLOCKS):
            return "handover"
        
        # Check if in feeding block and this nurse is a feeder
        if self.is_active_feeder and in_any_block(time_min, FEEDING_BLOCKS):
            return "feeding"
        
        # Check nurse round blocks
        if in_any_block(time_min, NURSE_ROUNDS_BLOCKS):
            return "rounding"
        
        # Check ad hoc window
        if in_block(time_min, AD_HOC_BLOCK):
            return "ad_hoc"
        
        # Default to station/idle
        return "station"
    
    def prepare_round_block(self, tick: int, time_min: int):
        """Prepare the nurse's visit plan for entering a new nurse round block.
        
        Distributes all patients in the caseload across ticks within the block.
        Uses proportional allocation with jitter to spread visits smoothly and
        avoid artificial spikes. Ensures all patients are visited exactly once.
        
        Called once when entering a new NURSE_ROUNDS_BLOCKS period.
        """
        bidx = block_index(time_min, NURSE_ROUNDS_BLOCKS)
        if bidx is None:
            return
        
        # Only prepare if not already prepared for this block
        if self.round_block_started_at_tick == tick:
            return
        
        # Gather all patients in caseload and shuffle
        caseload_patients = []
        for rid in self.caseload_rooms:
            caseload_patients.extend(self.model.get_patients_in_room(rid))
        self.model.rng.shuffle(caseload_patients)
        
        # Distribute patients across ticks with proportional + jitter allocation
        block_start_tick = NURSE_ROUNDS_BLOCKS[bidx][0] // self.model.config.dt_min
        block_end_tick = NURSE_ROUNDS_BLOCKS[bidx][1] // self.model.config.dt_min
        
        self.round_visits_per_tick = {}
        remaining_patients = list(caseload_patients)
        
        for current_tick in range(block_start_tick, block_end_tick):
            ticks_left = block_end_tick - current_tick
            patients_left = len(remaining_patients)
            
            if patients_left == 0:
                break
            
            # Greedy allocation: ensure all visits complete by block end
            n_to_visit = max(1, math.ceil(patients_left / ticks_left))
            # Add small jitter (-1, 0, or +1) to reduce determinism
            jitter = self.model.rng.randint(-1, 2)
            n_to_visit = max(1, min(n_to_visit + jitter, patients_left))
            
            # Randomly select patients for this tick
            visits_this_tick = self.model.rng.sample(remaining_patients, k=n_to_visit)
            self.round_visits_per_tick[current_tick] = visits_this_tick
            
            # Remove assigned patients from remaining
            for pid in visits_this_tick:
                remaining_patients.remove(pid)
        
        self.round_block_started_at_tick = tick
    
    def prepare_feeding_block(self, tick: int, time_min: int):
        """Prepare the nursing feeding visit plan for entering a new feeding block.
        
        Only called if this nurse is an active feeder. Gets assignment from model
        and distributes across ticks with proportional + jitter allocation.
        
        Called once when entering a new FEEDING_BLOCKS period (if feeder).
        """
        if not self.is_active_feeder:
            return
        
        bidx = block_index(time_min, FEEDING_BLOCKS)
        if bidx is None:
            return
        
        # Only prepare if not already prepared for this block
        if self.feeding_block_started_at_tick == tick:
            return
        
        # Get assignment from model (which caches and distributes across feeders)
        assigned = self.model.get_feeding_assignment_for_nurse(self.unique_id, bidx)
        self.model.rng.shuffle(assigned)
        
        # Distribute feedings across ticks with proportional + jitter allocation
        block_start_tick = FEEDING_BLOCKS[bidx][0] // self.model.config.dt_min
        block_end_tick = FEEDING_BLOCKS[bidx][1] // self.model.config.dt_min
        
        self.feeding_visits_per_tick = {}
        remaining_patients = list(assigned)
        
        for current_tick in range(block_start_tick, block_end_tick):
            ticks_left = block_end_tick - current_tick
            patients_left = len(remaining_patients)
            
            if patients_left == 0:
                break
            
            # Greedy allocation: ensure all feedings complete by block end
            n_to_feed = max(1, math.ceil(patients_left / ticks_left))
            # Add small jitter
            jitter = self.model.rng.randint(-1, 2)
            n_to_feed = max(1, min(n_to_feed + jitter, patients_left))
            
            # Randomly select patients for this tick
            feedings_this_tick = self.model.rng.sample(remaining_patients, k=n_to_feed)
            self.feeding_visits_per_tick[current_tick] = feedings_this_tick
            
            # Remove assigned patients from remaining
            for pid in feedings_this_tick:
                remaining_patients.remove(pid)
        
        self.feeding_block_started_at_tick = tick
    
    def _handle_rounding(self, tick: int, time_min: int):
        """Handle nurse round visits during scheduled blocks (6-7 AM, 4-5 PM).
        
        Executes visits pre-assigned to this tick from the block's visit plan.
        Visits are pre-distributed across ticks with jitter to ensure smooth
        temporal distribution and avoid quota-based spikiness.
        
        This approach ensures:
        - All patients in caseload are visited exactly once per block
        - Smooth temporal distribution with randomized timing
        - No artificial spikes from deterministic allocation
        """
        bidx = block_index(time_min, NURSE_ROUNDS_BLOCKS)
        if bidx is None:
            return
        
        block_start_tick = NURSE_ROUNDS_BLOCKS[bidx][0] // self.model.config.dt_min
        
        # Prepare visit plan on first tick of block
        if tick == block_start_tick:
            self.prepare_round_block(tick, time_min)
        
        # Execute visits scheduled for this tick (from pre-assigned plan)
        if tick in self.round_visits_per_tick:
            for pid in self.round_visits_per_tick[tick]:
                self.model.record_contact(
                    tick=tick,
                    actor_id=self.unique_id,
                    target_id=pid,
                    event_type="nurse_round",
                )
    
    def _handle_feeding(self, tick: int, time_min: int):
        """Handle feeding duties during scheduled feeding blocks.
        
        Only called if this nurse is designated as a feeder. Executes feedings
        pre-assigned to this tick from the block's feeding plan. Feedings are
        pre-distributed across ticks with jitter to ensure smooth timing.
        
        This approach ensures:
        - Feeders' workload is realistic (30-50% of patients per block)
        - No extreme dominance of feeder nodes in network
        - Smooth temporal distribution with randomized timing
        """
        if not self.is_active_feeder:
            return
        
        bidx = block_index(time_min, FEEDING_BLOCKS)
        if bidx is None:
            return
        
        block_start_tick = FEEDING_BLOCKS[bidx][0] // self.model.config.dt_min
        
        # Prepare feeding plan on first tick of block
        if tick == block_start_tick:
            self.prepare_feeding_block(tick, time_min)
        
        # Execute feedings scheduled for this tick (from pre-assigned plan)
        if tick in self.feeding_visits_per_tick:
            for pid in self.feeding_visits_per_tick[tick]:
                self.model.record_contact(
                    tick=tick,
                    actor_id=self.unique_id,
                    target_id=pid,
                    event_type="feeding",
                )
    
    def _handle_handover(self, tick: int, time_min: int):
        """Handle shift handover interactions at nurse station.
        
        During handover periods (6:55-7:05 AM, 3:55-4:05 PM), nurses gather at the station.
        Nurses probabilistically initiate interactions with each other or doctors.
        
        IMPROVED: Now agent-driven instead of purely model-managed.
        Uses proper block_index() tracking to identify block transitions and reset
        the initiated flag reliably for both morning and afternoon periods.
        """
        # Check if we're in a handover period using proper block indexing
        bidx = block_index(time_min, HANDOVER_BLOCKS)
        if bidx is None:
            # Left a handover block; reset flag for next block
            if self.handover_block_idx != -1:
                self.handover_block_idx = -1
                self.handover_initiated_this_block = False
            return
        
        # Entering a new handover block; reset flag
        if bidx != self.handover_block_idx:
            self.handover_block_idx = bidx
            self.handover_initiated_this_block = False
        
        # Avoid excessive handover interactions per block
        # Each nurse initiates at most once per handover period
        if self.handover_initiated_this_block:
            return
        
        # Moderate probability of initiating a handover contact
        if self.model.rng.random() > 0.6:
            return
        
        # Nurse can interact with another nurse or a doctor
        interactant_option = self.model.rng.choice(["nurse", "doctor"])
        
        if interactant_option == "nurse":
            other_nurses = [n for n in self.model.nurses if n.unique_id != self.unique_id]
            if other_nurses:
                other = self.model.rng.choice(other_nurses)
                # Check if we haven't recently contacted this nurse (undirected pair)
                if not self.model.is_recent_contact(self.unique_id, other.unique_id, tick, window_ticks=5):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=other.unique_id,
                        event_type="nurse_station",
                    )
                    self.handover_initiated_this_block = True
        else:  # doctor
            if self.model.doctors:
                doctor = self.model.rng.choice(self.model.doctors)
                # Check if we haven't recently contacted this doctor (undirected pair)
                if not self.model.is_recent_contact(self.unique_id, doctor.unique_id, tick, window_ticks=5):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=doctor.unique_id,
                        event_type="nurse_station",
                    )
                    self.handover_initiated_this_block = True
    
    def _handle_ad_hoc(self, tick: int, time_min: int):
        """Handle occasional unscheduled patient visits during daytime window (9 AM - 3 PM).
        
        With low probability, a nurse may initiate a spontaneous visit to a random patient
        from their caseload. Avoids excessive repeated contacts to the same patient.
        """
        if not in_block(time_min, AD_HOC_BLOCK):
            return
        
        # Low probability of ad-hoc visit initiation
        if self.model.rng.random() > (self.model.config.p_ad_hoc_tick * 0.3):
            return
        
        # Pick a random patient from my caseload to visit
        if self.caseload_rooms:
            caseload_patients = []
            for rid in self.caseload_rooms:
                caseload_patients.extend(self.model.get_patients_in_room(rid))
            if caseload_patients:
                pid = self.model.rng.choice(caseload_patients)
                # Avoid excessive repeated ad-hoc contacts to same patient
                if not self.model.is_recent_contact(self.unique_id, pid, tick, window_ticks=24):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=pid,
                        event_type="ad_hoc",
                    )


class DoctorAgent(BaseHospitalAgent):
    """
    Doctor agent with explicit state machine and own step() behavior.
    
    State machine:
    - "rounding": Visiting assigned patients during doctor round blocks (7-8 AM, 3-4 PM).
    - "handover": At nurse station during shift handover times.
    - "ad_hoc": Occasional unscheduled patient visits during daytime.
    - "station": At doctor's office or nurse station for administrative work.
    - "idle": Off-duty or between assignments.
    
    IMPROVEMENTS: Like NurseAgent, doctors now maintain persistent block-level visit plans
    that are precomputed at each round block start and distributed across ticks with jitter.
    This eliminates quota-based spikiness and creates more realistic temporal behavior.
    
    Attributes:
        panel_patients: List of patient IDs this doctor is responsible for.
        current_state: Current state (one of the above).
        
        Block planning (persistent per block):
        round_visits_per_tick: Dict[int, List[str]] mapping tick to patients to visit.
        round_block_started_at_tick: Tick when current plan was created.
        
        handover_block_idx: Index of the current handover block (-1 if not in one).
        handover_initiated_this_block: Whether initiated contact in current block.
    """
    def __init__(self, model: Model, unique_id: str):
        super().__init__(model, unique_id, "doctor")
        self.panel_patients: list[str] = []  # List of patient IDs this doctor visits
        
        # State machine
        self.current_state: str = "idle"
        
        # Block-level persistent visit plan (tick -> list of patients)
        self.round_visits_per_tick: dict[int, list[str]] = {}  # Pre-assigned visits per tick
        self.round_block_started_at_tick: int = -1  # When current plan was created
        
        # Handover state (track block to reset properly)
        self.handover_block_idx: int = -1  # Current handover block index
        self.handover_initiated_this_block: bool = False
    
    def step(self):
        """Execute one tick of doctor behavior based on current time and state."""
        time_min = self.model.get_current_time_min()
        tick = self.model.current_tick
        
        # Determine what state we should be in based on current time
        self.current_state = self._get_current_state(time_min)
        
        # Execute behavior appropriate to current state
        if self.current_state == "rounding":
            self._handle_rounding(tick, time_min)
        elif self.current_state == "handover":
            self._handle_handover(tick, time_min)
        elif self.current_state == "ad_hoc":
            self._handle_ad_hoc(tick, time_min)
        # station and idle states don't generate proactive contacts
    
    def _get_current_state(self, time_min: int) -> str:
        """Determine the doctor's current state based on time of day.
        
        Priority order: rounding > handover > ad_hoc > station/idle
        """
        # Check doctor round blocks
        if in_any_block(time_min, DOCTOR_BLOCKS):
            return "rounding"
        
        # Check handover
        if in_any_block(time_min, HANDOVER_BLOCKS):
            return "handover"
        
        # Check ad hoc window
        if in_block(time_min, AD_HOC_BLOCK):
            return "ad_hoc"
        
        # Default to station/idle
        return "station"
    
    def prepare_round_block(self, tick: int, time_min: int):
        """Prepare the doctor's visit plan for entering a new doctor round block.
        
        Distributes all panel patients across ticks within the block with
        proportional allocation and jitter to smooth the temporal distribution.
        Ensures all patients are visited exactly once per block.
        
        Called once when entering a new DOCTOR_BLOCKS period.
        """
        bidx = block_index(time_min, DOCTOR_BLOCKS)
        if bidx is None:
            return
        
        # Only prepare if not already prepared for this block
        if self.round_block_started_at_tick == tick:
            return
        
        # Copy and shuffle panel patients
        panel_copy = list(self.panel_patients)
        self.model.rng.shuffle(panel_copy)
        
        # Distribute patients across ticks with proportional + jitter allocation
        block_start_tick = DOCTOR_BLOCKS[bidx][0] // self.model.config.dt_min
        block_end_tick = DOCTOR_BLOCKS[bidx][1] // self.model.config.dt_min
        
        self.round_visits_per_tick = {}
        remaining_patients = list(panel_copy)
        
        for current_tick in range(block_start_tick, block_end_tick):
            ticks_left = block_end_tick - current_tick
            patients_left = len(remaining_patients)
            
            if patients_left == 0:
                break
            
            # Greedy allocation: ensure all visits complete by block end
            n_to_visit = max(1, math.ceil(patients_left / ticks_left))
            # Add small jitter to reduce determinism
            jitter = self.model.rng.randint(-1, 2)
            n_to_visit = max(1, min(n_to_visit + jitter, patients_left))
            
            # Randomly select patients for this tick
            visits_this_tick = self.model.rng.sample(remaining_patients, k=n_to_visit)
            self.round_visits_per_tick[current_tick] = visits_this_tick
            
            # Remove assigned patients from remaining
            for pid in visits_this_tick:
                remaining_patients.remove(pid)
        
        self.round_block_started_at_tick = tick
    
    def _handle_rounding(self, tick: int, time_min: int):
        """Handle doctor round visits during scheduled blocks (7-8 AM, 3-4 PM).
        
        Executes visits pre-assigned to this tick from the block's visit plan.
        Visits are pre-distributed across ticks with jitter to ensure smooth
        temporal distribution and avoid quota-based spikiness.
        
        This approach ensures:
        - All patients in panel are visited exactly once per block
        - Smooth temporal distribution with randomized timing
        - No artificial spikes from deterministic allocation
        """
        bidx = block_index(time_min, DOCTOR_BLOCKS)
        if bidx is None:
            return
        
        block_start_tick = DOCTOR_BLOCKS[bidx][0] // self.model.config.dt_min
        
        # Prepare visit plan on first tick of block
        if tick == block_start_tick:
            self.prepare_round_block(tick, time_min)
        
        # Execute visits scheduled for this tick (from pre-assigned plan)
        if tick in self.round_visits_per_tick:
            for pid in self.round_visits_per_tick[tick]:
                self.model.record_contact(
                    tick=tick,
                    actor_id=self.unique_id,
                    target_id=pid,
                    event_type="doctor_round",
                )
    
    def _handle_handover(self, tick: int, time_min: int):
        """Handle shift handover interactions at nurse station.
        
        During handover periods, doctors may interact with nurses and other doctors.
        
        IMPROVED: Now agent-driven instead of purely model-managed.
        Uses proper block_index() tracking to identify block transitions and reset
        the initiated flag reliably for both morning and afternoon periods.
        """
        # Check if we're in a handover period using proper block indexing
        bidx = block_index(time_min, HANDOVER_BLOCKS)
        if bidx is None:
            # Left a handover block; reset flag for next block
            if self.handover_block_idx != -1:
                self.handover_block_idx = -1
                self.handover_initiated_this_block = False
            return
        
        # Entering a new handover block; reset flag
        if bidx != self.handover_block_idx:
            self.handover_block_idx = bidx
            self.handover_initiated_this_block = False
        
        # Avoid excessive handover interactions per block (doctors are less vocal than nurses)
        if self.handover_initiated_this_block:
            return
        
        # Doctors have slightly lower initiation probability than nurses
        if self.model.rng.random() > 0.5:
            return
        
        # Doctor can interact with another doctor or a nurse
        interactant_option = self.model.rng.choice(["doctor", "nurse"])
        
        if interactant_option == "doctor":
            other_doctors = [d for d in self.model.doctors if d.unique_id != self.unique_id]
            if other_doctors:
                other = self.model.rng.choice(other_doctors)
                # Check if we haven't recently contacted this doctor (undirected pair)
                if not self.model.is_recent_contact(self.unique_id, other.unique_id, tick, window_ticks=5):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=other.unique_id,
                        event_type="nurse_station",
                    )
                    self.handover_initiated_this_block = True
        else:  # nurse
            if self.model.nurses:
                nurse = self.model.rng.choice(self.model.nurses)
                # Check if we haven't recently contacted this nurse (undirected pair)
                if not self.model.is_recent_contact(self.unique_id, nurse.unique_id, tick, window_ticks=5):
                    self.model.record_contact(
                        tick=tick,
                        actor_id=self.unique_id,
                        target_id=nurse.unique_id,
                        event_type="nurse_station",
                    )
                    self.handover_initiated_this_block = True
    
    def _handle_ad_hoc(self, tick: int, time_min: int):
        """Handle occasional unscheduled patient visits during daytime window (9 AM - 3 PM).
        
        With low probability, a doctor may initiate a spontaneous visit to a random patient
        from their panel. Avoids excessive repeated contacts to the same patient.
        """
        if not in_block(time_min, AD_HOC_BLOCK):
            return
        
        # Low probability of ad-hoc visit initiation (doctors initiate less than nurses)
        if self.model.rng.random() > (self.model.config.p_ad_hoc_tick * 0.15):
            return
        
        if not self.panel_patients:
            return
        
        pid = self.model.rng.choice(self.panel_patients)
        # Avoid excessive repeated ad-hoc contacts to same patient
        if not self.model.is_recent_contact(self.unique_id, pid, tick, window_ticks=24):
            self.model.record_contact(
                tick=tick,
                actor_id=self.unique_id,
                target_id=pid,
                event_type="ad_hoc",
            )


# =========================================================
# 5) MESA MODEL: HospitalContactModel - Core simulation engine
# =========================================================
class HospitalContactModel(Model):
    """
    Mesa Model for hospital contact network simulation.
    
    Generates contact events between patients, nurses, and doctors during a single day,
    based on scheduled activities (rounds, feeding, handovers) and random interactions.
    No infection dynamics are simulated; this model only tracks who contacts whom.
    
    REFACTORED: This version is more agent-centric. Nurse and doctor agents
    run their own step() methods and decide behavior based on state machines.
    The model coordinates the environment, provides helper methods to agents,
    manages agent scheduling, and handles remaining centralized events
    (roommate interactions, nurse station general gathering).
    """
    def __init__(self, config: SimConfig):
        super().__init__()
        self.config = config
        self.rng = random.Random(config.seed)

        # Core maps for room management
        self.room_capacity_map: dict[str, int] = self._build_room_capacity_map()
        self.room_occupants: dict[str, list[str]] = {rid: [] for rid in self.room_capacity_map}

        # Agent registries: maintain lists of each agent type for quick iteration
        self.patients: list[PatientAgent] = []
        self.nurses: list[NurseAgent] = []
        self.doctors: list[DoctorAgent] = []
        self.agent_index: dict[str, BaseHospitalAgent] = {}  # Fast lookup by unique_id

        # Simulation logs
        self.visit_events: list[dict] = []  # All recorded contact events

        # Daily bookkeeping
        self.current_tick = 0
        self.current_time_min = 0

        # Track feeding assignments: feeding block index -> {nurse_id: [patient_ids]}
        self._feeding_block_assignments: dict[int, dict[str, list[str]]] = {}

        # Track whether roommate event has occurred in a given room during a given hour
        # Ensures at most one roommate event per room per hour
        self._room_hour_triggered: set[tuple[str, int]] = set()
        
        # Short-term contact tracking to avoid excessive repeated interactions.
        # Key: (actor_id, target_id), Value: last_tick of contact
        # Used to prevent duplicate ad-hoc or hand-over contacts within a time window.
        self._recent_contacts: dict[tuple[str, str], int] = {}

        # Pre-sample random times during daytime when nurses gather at nurse station
        self._random_nurse_station_ticks = self._sample_random_nurse_station_ticks()

        # Initialize all agents and establish assignments
        self._init_agents()  # Create patient, nurse, and doctor instances
        self._assign_patients_to_rooms_deterministic()  # Fill rooms by capacity
        self._assign_nurse_room_caseloads()  # Each nurse gets adjacent rooms (round-robin)
        self._assign_doctor_panels()  # Each doctor gets subset of patients (round-robin)
        self._assign_daily_feeders()  # Randomly select 2 nurses as feeders for the day

    def _build_room_capacity_map(self) -> dict[str, int]:
        """Create mapping of room IDs to bed capacities from config constants."""
        assert len(ROOM_CAPACITY_SPEC) == self.config.n_rooms
        return {f"room_{i}": cap for i, cap in enumerate(ROOM_CAPACITY_SPEC)}

    def _init_agents(self):
        """Initialize all patient, nurse, and doctor agents and add to agent index."""
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
        """Assign patients to rooms according to room capacity in order (deterministic)."""
        patient_iter = iter(self.patients)
        for room_id, capacity in self.room_capacity_map.items():
            for _ in range(capacity):
                patient = next(patient_iter)
                patient.room_id = room_id
                self.room_occupants[room_id].append(patient.unique_id)

        total_capacity = sum(self.room_capacity_map.values())
        assert total_capacity == self.config.n_patients, "Room capacity must exactly match patient count"

    def _assign_nurse_room_caseloads(self):
        """Distribute rooms among nurses via round-robin assignment."""
        room_ids = sorted(self.room_capacity_map.keys(), key=lambda x: int(x.split("_")[1]))
        for i, room_id in enumerate(room_ids):
            nurse = self.nurses[i % len(self.nurses)]
            nurse.caseload_rooms.append(room_id)

    def _assign_doctor_panels(self):
        """Distribute patients among doctors via round-robin assignment."""
        patient_ids = [p.unique_id for p in self.patients]
        for i, pid in enumerate(patient_ids):
            doctor = self.doctors[i % len(self.doctors)]
            doctor.panel_patients.append(pid)

    def _assign_daily_feeders(self):
        """Randomly select two nurses to serve as feeders for the day (without replacement)."""
        feeder_indices = self.rng.sample(range(len(self.nurses)), k=2)
        for i, nurse in enumerate(self.nurses):
            nurse.is_active_feeder = i in feeder_indices

    def _sample_random_nurse_station_ticks(self) -> set[int]:
        """Pre-sample random times during daytime when nurses gather at nurse station.
        
        Used in addition to scheduled handover periods for unscheduled nurse interactions.
        """
        daytime_tick_start = AD_HOC_BLOCK[0] // self.config.dt_min
        daytime_tick_end = AD_HOC_BLOCK[1] // self.config.dt_min
        all_daytime_ticks = list(range(daytime_tick_start, daytime_tick_end))

        k = min(self.config.nurse_station_random_ticks_per_day, len(all_daytime_ticks))
        return set(self.rng.sample(all_daytime_ticks, k=k))

    # =========================================================
    # Helper Methods for Agents to Call
    # =========================================================
    
    def get_current_time_min(self) -> int:
        """Return current simulation time in minutes since start of day."""
        return self.current_time_min
    
    def get_patients_in_room(self, room_id: str) -> list[str]:
        """Return list of patient IDs currently in a given room."""
        return self.room_occupants.get(room_id, [])
    
    def record_contact(
        self,
        tick: int,
        actor_id: str,
        target_id: str,
        event_type: str,
        duration_min: int = DURATION_MIN_DEFAULT,
    ):
        """Record a single contact event initiated by an agent.
        
        Called from agent step() methods to log contacts they generate.
        Automatically looks up room from target patient and determines actor/target types.
        Also updates the short-term contact tracking dictionary using canonical
        (undirected) pair keys to suppress repeated interactions between any pair.
        """
        actor = self.agent_index[actor_id]
        target = self.agent_index[target_id]
        time_min = tick_to_time_min(tick, self.config.dt_min)
        
        # Get room: if target is patient, use their room; otherwise nurse_station
        if target.agent_type == "patient":
            room_id = target.room_id
        else:
            room_id = "nurse_station"

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
        
        # Update short-term contact tracking using canonical (sorted) pair key.
        # This ensures (actor_id, target_id) and (target_id, actor_id) map to the same entry,
        # treating the pair as undirected for suppression purposes.
        contact_pair = tuple(sorted([actor_id, target_id]))
        self._recent_contacts[contact_pair] = tick
    
    def is_tick_in_block(self, tick: int, block: tuple[int, int]) -> bool:
        """Check if a given tick falls within a time block [start, end)."""
        time_min = tick_to_time_min(tick, self.config.dt_min)
        return in_block(time_min, block)
    
    def is_recent_contact(self, actor_id: str, target_id: str, current_tick: int, window_ticks: int = 12) -> bool:
        """Check if two agents have had contact recently (within window_ticks).
        
        This prevents excessive repeated interactions between the same pair,
        which would be unrealistic and create artificial star topologies.
        Uses canonical (sorted) pair keys to treat interactions as undirected:
        is_recent_contact(A, B) returns True iff (A, B) or (B, A) contacted recently.
        
        Args:
            actor_id: First agent
            target_id: Second agent
            current_tick: Current simulation tick
            window_ticks: Number of ticks to look back (default 12 = 1 hour)
        
        Returns:
            True if the pair had contact within the window, False otherwise.
        """
        # Use canonical pair key (sorted) to treat pair as undirected
        contact_pair = tuple(sorted([actor_id, target_id]))
        if contact_pair not in self._recent_contacts:
            return False
        
        last_contact_tick = self._recent_contacts[contact_pair]
        return (current_tick - last_contact_tick) < window_ticks
    
    def get_feeding_assignment_for_nurse(self, nurse_id: str, bidx: int) -> list[str]:
        """Get the list of patients assigned to a given nurse for a specific feeding block.
        
        Called by feeder nurses in their _handle_feeding() method.
        Assignments are pre-generated on first call and cached.
        
        IMPROVEMENT: Now avoids making feeders unrealistically dominant.
        Each feeder gets roughly 35-50% of patients (not 60-80% as before),
        distributed among fewer total patients per block.
        """
        # If assignments for this block haven't been generated yet, create them
        if bidx not in self._feeding_block_assignments:
            # Reduce feeding coverage to be more realistic and less nurse-centric
            # instead of 60-80%, use smaller subset: 30-50% of patients
            coverage = self.rng.uniform(0.30, 0.50)
            n_target = max(1, int(round(self.config.n_patients * coverage)))
            
            patient_ids = [p.unique_id for p in self.patients]
            selected = self.rng.sample(patient_ids, k=n_target)
            
            # Split between the two active feeders more evenly
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
    # Model-Level Event Generation (not agent-driven)
    # =========================================================
    
    def _generate_roommate_events(self, tick: int, time_min: int):
        """Generate interactions between roommates once per hour per room.
        
        Evaluates at the start of each hour (xx:00) for each room with 2+ patients.
        At most one roommate event per room per hour.
        
        Kept at model level: Roommates interacting is not driven by individual agent
        behavior; it's a passive background event that both agents experience similarly.
        """
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
                # Record as a model-initiated event
                time_min_val = tick_to_time_min(tick, self.config.dt_min)
                event = {
                    "run_id": self.config.run_id,
                    "tick": tick,
                    "time_min": time_min_val,
                    "time_str": minutes_to_hhmm(time_min_val),
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
        """Generate background staff interactions at the nurse station.
        
        Creates occasional random staff-to-staff contacts that happen at the nurse station.
        This is a reduced, background-only version now that agents handle their own
        handover and ad-hoc interactions.
        
        The model now generates only rare, unplanned encounters between staff members
        who happen to pass through the nurse station, rather than scheduled gatherings.
        Scheduled handover contacts are now agent-driven (see NurseAgent and DoctorAgent
        _handle_handover() methods).
        
        Kept at model level: These are incidental, background interactions that don't
        fit neatly into any individual agent's planned activities. They represent the
        natural emergent effect of all staff moving through a shared space.
        """
        # Only generate background random nurse station events during daytime
        # Don't replicate what agents are already doing during handover
        is_random_daytime_tick = tick in self._random_nurse_station_ticks

        if not is_random_daytime_tick:
            return
        
        # Generate occasional nurse-nurse random encounters (very low rate)
        if self.rng.random() < 0.3 and len(self.nurses) >= 2:
            n1, n2 = self.rng.sample(self.nurses, k=2)
            if not self.is_recent_contact(n1.unique_id, n2.unique_id, tick, window_ticks=6):
                time_min_val = tick_to_time_min(tick, self.config.dt_min)
                event = {
                    "run_id": self.config.run_id,
                    "tick": tick,
                    "time_min": time_min_val,
                    "time_str": minutes_to_hhmm(time_min_val),
                    "actor_id": n1.unique_id,
                    "actor_type": "nurse",
                    "target_id": n2.unique_id,
                    "target_type": "nurse",
                    "room_id": "nurse_station",
                    "event_type": "nurse_station",
                    "duration_min": DURATION_MIN_DEFAULT,
                }
                self.visit_events.append(event)
                # Use canonical pair key for undirected tracking
                contact_pair = tuple(sorted([n1.unique_id, n2.unique_id]))
                self._recent_contacts[contact_pair] = tick

    def step(self):
        """Execute one tick of the simulation.
        
        Now with a more agent-centric approach:
        1. Update current time.
        2. Step all agents (patients, nurses, doctors). Agents decide their behavior
           and call model.record_contact() to log events.
        3. Generate model-level events that aren't driven by individual agents
           (roommate interactions, nurse station gatherings).
        """
        tick = self.current_tick
        time_min = tick_to_time_min(tick, self.config.dt_min)
        self.current_time_min = time_min

        # Step all agents: they decide their behavior and generate contacts
        for patient in self.patients:
            patient.step()
        
        for nurse in self.nurses:
            nurse.step()
        
        for doctor in self.doctors:
            doctor.step()
        
        # Generate model-level events (not initiated by individual agents)
        self._generate_roommate_events(tick, time_min)
        self._generate_nurse_station_events(tick, time_min)

        self.current_tick += 1


# =========================================================
# 7) SIMULATION EXECUTION: Run one complete day
# =========================================================
def run_simulation(config: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one complete day of hospital contact simulation.
    
    Returns:
        visit_df: Detailed log of all contact events
        agg_df: Aggregated contact pairs with frequencies
        summary_df: Summary statistics for the simulation run
    """
    model = HospitalContactModel(config)

    # Step through all ticks in a 24-hour day
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
# 8) DATA AGGREGATION & SUMMARY: Process simulation results
# =========================================================
def build_aggregated_edges(visit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate contact events into unique undirected edges with frequency counts.
    
    Canonically orders each pair (alphabetically by ID) to merge bidirectional contacts.
    """
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
    """Return the canonical 2-letter edge type code (e.g., 'NP', 'PP', 'DD')."""
    pair = sorted([actor_type[0].upper(), target_type[0].upper()])
    return "".join(pair)  # e.g. ['N','P'] -> 'NP'


def build_run_summary(config: SimConfig, visit_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary statistics table for the simulation run.
    
    Computes total events, edge type counts, network metrics (degree, weighted degree),
    and identifies top nodes by centrality.
    """
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


# =========================================================
# 10) CSV EXPORT: Save results to output files
# =========================================================

def export_csvs(config: SimConfig, visit_df: pd.DataFrame, agg_df: pd.DataFrame, summary_df: pd.DataFrame, run_output_dir: str | None = None):
    """Export simulation results to CSV files in the run-specific output directory.
    
    Creates three CSV files in the run_output_dir: visit_log.csv, aggregated_edges.csv, 
    and run_summary.csv. Each run gets its own folder, so previous results are never 
    overwritten.
    
    Args:
        config: Simulation configuration
        visit_df: Detailed contact event log
        agg_df: Aggregated contact pairs with frequencies
        summary_df: Summary statistics for the run
        run_output_dir: Run-specific output directory. If None, uses config.output_dir.
    
    Returns:
        Tuple of (visit_path, agg_path, summary_path) for the exported CSV files.
    """
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
# 9) VISUALIZATION: Network graphs and time series plots
# =========================================================
def plot_network(config: SimConfig, agg_df: pd.DataFrame, out_path: str):
    """Create and save a network visualization using spring layout.
    
    Nodes are colored by role (patient=blue, nurse=orange, doctor=green).
    Edge width represents contact frequency.
    """
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
    """Create and save a time series plot of contact events throughout the day.
    
    Shows total events per tick, and breakdown by Patient-Nurse and Patient-Doctor.
    """
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
    """Create and save degree distribution histograms.
    
    Shows both unweighted and weighted (by contact count) degree distributions,
    separated by agent role.
    """
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
    """Generate and save all three visualization figures to the run-specific directory.
    
    Creates network graph, time series plot, and degree distribution histogram.
    All figures are saved to run_output_dir/figures/ to keep each run's outputs
    completely isolated from previous runs.
    
    Args:
        config: Simulation configuration
        visit_df: Detailed contact event log
        agg_df: Aggregated contact pairs with frequencies
        run_output_dir: Run-specific output directory. If None, uses config.output_dir.
    
    Returns:
        Tuple of (network_path, timeseries_path, degree_hist_path) for the PNG files.
    """
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
# 11) COMMAND-LINE INTERFACE & MAIN ENTRY POINT
# =========================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for run seed and run ID."""
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
    """Main entry point: configure, run simulation, and export results to run-specific folder."""
    args = parse_args()

    run_id = args.run_id or str(int(datetime.utcnow().timestamp()))
    config = SimConfig(seed=args.seed, run_id=run_id)
    
    # Build a unique output directory for this run to avoid overwriting previous results.
    # The directory name includes timestamp, seed, and run_id to uniquely identify the run.
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
