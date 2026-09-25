"""
Program database for OpenEvolve
"""

import base64
import json
import logging
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields

# FileLock removed - no longer needed with threaded parallel processing
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from openevolve.config import DatabaseConfig
from openevolve.utils.code_utils import calculate_edit_distance
from openevolve.utils.metrics_utils import get_fitness_score, safe_numeric_average

logger = logging.getLogger(__name__)


def _safe_sum_metrics(metrics: Dict[str, Any]) -> float:
    """Safely sum only numeric metric values, ignoring strings and other types"""
    numeric_values = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) if numeric_values else 0.0


def _safe_avg_metrics(metrics: Dict[str, Any]) -> float:
    """Safely calculate average of only numeric metric values"""
    numeric_values = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) / max(1, len(numeric_values)) if numeric_values else 0.0


def is_feasible(program: "Program", feasibility_metric: Optional[str]) -> bool:
    """Whether `program` may be bred from.

    A missing metric is feasible: a program that has not been measured on the
    constraint is not thereby in violation of it, and with no
    `feasibility_metric` configured every program is feasible, which is upstream
    exactly. Free of side effects.
    """
    if feasibility_metric is None:
        return True
    value = (program.metrics or {}).get(feasibility_metric)
    if value is None:
        return True
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return True


def violation_of(program: "Program", violation_metric: Optional[str]) -> float:
    """How far `program` is from feasible; lower is closer.

    Absent or unparseable is the worst, so a program carrying no measurement
    never outranks one that does when the fallback is choosing.
    """
    if violation_metric is None:
        return 0.0
    value = (program.metrics or {}).get(violation_metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def tiebreak_of(program: "Program", tiebreak_metric: Optional[str]) -> Optional[float]:
    """`program`'s tiebreak value, or None when it carries no usable one.

    None rather than an extreme, so the caller can say what an absent
    measurement means rather than have a sentinel decide it. Free of side
    effects.
    """
    if tiebreak_metric is None:
        return None
    value = (program.metrics or {}).get(tiebreak_metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def breedable(programs: list, feasibility_metric: Optional[str]) -> list:
    """The members of `programs` worth breeding from.

    The feasible ones when there are any; otherwise all of them, so a scope that
    has not yet produced a feasible program still yields a parent (ranked by
    violation elsewhere) rather than nothing. Identity when unconfigured.
    """
    if feasibility_metric is None or not programs:
        return programs
    feasible = [p for p in programs if is_feasible(p, feasibility_metric)]
    return feasible or programs


def lane_group_of(program: "Program", lane_metric: Optional[str]) -> Any:
    """The program's lane-group key under `lane_metric` (a rounded-int bucket).

    Missing/unparseable metric -> -1; no lane_metric configured -> None. Free of
    any ProgramDatabase state so evaluation workers, which hold no database, can
    group programs the same way the controller does.
    """
    if lane_metric is None:
        return None
    value = (program.metrics or {}).get(lane_metric)
    if value is None:
        return -1
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return -1


def _group_score(program: "Program") -> float:
    """Combined score for within-group ranking; missing sorts to the bottom."""
    value = (program.metrics or {}).get("combined_score") if program.metrics else None
    return float(value) if value is not None else float("-inf")


def dedup_exemplars(programs: list, key: Optional[str]) -> list:
    """`programs` with one member per distinct `metadata[key]`, order preserved.

    The exemplar lists are what the generator is shown, and it copies what it
    sees; a population holding many copies of one program would otherwise spend
    every slot on that one. A program carrying no value under `key` is never
    collapsed -- an unidentified program is not thereby a duplicate of another
    unidentified one. Identity when `key` is None, which is upstream exactly.
    """
    if key is None:
        return programs
    seen: set = set()
    out = []
    for program in programs:
        value = (program.metadata or {}).get(key)
        if value is None:
            out.append(program)
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(program)
    return out


def rank_exemplars(
    programs: list,
    feasibility_metric: Optional[str],
    violation_metric: Optional[str],
    feature_dimensions: Optional[List[str]] = None,
    score=None,
) -> list:
    """`programs` ordered as exemplars: feasible first, by fitness.

    Selection already ranks feasibility first (`_breedable_ids`, `_is_better`),
    but the lists a program is *displayed* in were ordered on fitness alone, so
    an infeasible program that outscored its whole island headed every prompt
    written there. Infeasible members keep their place in the list -- their
    ideas are the reason the prompt shows near-misses at all -- but they sort
    after every feasible one, by violation ascending then fitness.

    `score` is the fitness function the call site already used, so each ranking
    keeps its own notion of best; the default is the feature-aware fitness.
    Identity -- today's plain sort -- when no `feasibility_metric` is
    configured, which is upstream exactly. Free of side effects.
    """
    if score is None:

        def score(program: "Program") -> float:
            return get_fitness_score(program.metrics, feature_dimensions)

    if feasibility_metric is None:
        return sorted(programs, key=score, reverse=True)

    def key(program: "Program"):
        feasible = is_feasible(program, feasibility_metric)
        return (
            0 if feasible else 1,
            0.0 if feasible else violation_of(program, violation_metric),
            -score(program),
        )

    return sorted(programs, key=key)


@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    changes_description: str = (
        ""  # compact program changes description (via LLM) stored per program
    )
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0  # Track which iteration this program was found

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Prompts
    prompts: Optional[Dict[str, Any]] = None

    # Artifact storage
    artifacts_json: Optional[str] = None  # JSON-serialized small artifacts
    artifact_dir: Optional[str] = None  # Path to large artifact files

    # Embedding vector for novelty rejection sampling
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # old DBs don't have changes_description (backward-compatibility)
        if "changes_description" not in data:
            metadata = data.get("metadata") or {}
            if isinstance(metadata, dict):
                data = {
                    **data,
                    "changes_description": metadata.get("changes_description")
                    or metadata.get("changes")
                    or "empty",
                }
            else:
                data = {**data, "changes_description": "empty"}

        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)


class ProgramDatabase:
    """
    Database for storing and sampling programs during evolution

    The database implements a combination of MAP-Elites algorithm and
    island-based population model to maintain diversity during evolution.
    It also tracks the absolute best program separately to ensure it's never lost.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config

        # In-memory program storage
        self.programs: Dict[str, Program] = {}

        # Per-island feature grids for MAP-Elites
        self.island_feature_maps: List[Dict[str, str]] = [{} for _ in range(config.num_islands)]

        # Handle both int and dict types for feature_bins
        if isinstance(config.feature_bins, int):
            self.feature_bins = max(
                config.feature_bins,
                int(pow(config.archive_size, 1 / len(config.feature_dimensions)) + 0.99),
            )
        else:
            # If dict, keep as is (we'll use feature_bins_per_dim instead)
            self.feature_bins = 10  # Default fallback for backward compatibility

        # Island populations
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]

        # Island management attributes
        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.migration_interval: int = getattr(config, "migration_interval", 10)  # Default to 10
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)  # Default to 0.1

        # Archive of elite programs
        self.archive: Set[str] = set()

        # Track the absolute best program separately
        self.best_program_id: Optional[str] = None

        # Track best program per island for proper island-based evolution
        self.island_best_programs: List[Optional[str]] = [None] * config.num_islands

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # Lanes fork: declared feature-axis domains (pin scaling range, no ratchet)
        # and lane-aware eviction metric. Both default to upstream behavior. Set
        # BEFORE load(), so a from-disk construction sees them (load()'s split-grid
        # check reads lane_split_grids).
        self.feature_domains: Dict[str, List[float]] = dict(
            getattr(config, "feature_domains", {}) or {}
        )
        self.lane_metric: Optional[str] = getattr(config, "lane_metric", None)
        # All inert unless feasibility_metric is set.
        self.dedup_key: Optional[str] = getattr(config, "dedup_key", None)
        self.reject_metric: Optional[str] = getattr(config, "reject_metric", None)
        self.feasibility_metric: Optional[str] = getattr(config, "feasibility_metric", None)
        self.feasibility_min_pool: int = int(getattr(config, "feasibility_min_pool", 0) or 0)
        self.violation_metric: Optional[str] = getattr(config, "violation_metric", None)
        # Tiebreak on exactly-equal fitness. Inert unless tiebreak_metric is set.
        self.tiebreak_metric: Optional[str] = getattr(config, "tiebreak_metric", None)
        self.tiebreak_lower_is_better: bool = bool(
            getattr(config, "tiebreak_lower_is_better", True)
        )
        # First-class record of every program removal (population cap, orphaning).
        self.eviction_journal: List[Dict[str, Any]] = []

        # Lanes fork: group-aware parent sampling, prompt scoping, and split grids.
        # All inert unless lane_metric is set.
        self.lane_sampling_gamma: Optional[float] = getattr(config, "lane_sampling_gamma", None)
        self.lane_rank_weight: float = getattr(config, "lane_rank_weight", 0.0)
        self.lane_rank_power: float = getattr(config, "lane_rank_power", 1.0)
        self.lane_prompt_scope: bool = getattr(config, "lane_prompt_scope", False)
        self.lane_cross_inspirations: int = getattr(config, "lane_cross_inspirations", 1)
        self.lane_split_grids: bool = getattr(config, "lane_split_grids", False)
        self.lane_group_migration: bool = getattr(config, "lane_group_migration", False)
        if (
            self.lane_sampling_gamma
            or self.lane_prompt_scope
            or self.lane_split_grids
            or self.lane_group_migration
        ) and self.lane_metric is None:
            logger.warning(
                "lane_sampling_gamma/lane_prompt_scope/lane_split_grids/lane_group_migration "
                "set without lane_metric; ignoring (group-aware behavior needs a metric to "
                "group by)"
            )
            self.lane_sampling_gamma = None
            self.lane_prompt_scope = False
            self.lane_split_grids = False
            self.lane_group_migration = False

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

        # Prompt log
        self.prompts_by_program: Dict[str, Dict[str, Dict[str, str]]] = None

        # Set random seed for reproducible sampling if specified
        if config.random_seed is not None:
            import random

            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")

        # Diversity caching infrastructure
        self.diversity_cache: Dict[int, Dict[str, Union[float, float]]] = (
            {}
        )  # hash -> {"value": float, "timestamp": float}
        self.diversity_cache_size: int = 1000  # LRU cache size
        self.diversity_reference_set: List[str] = (
            []
        )  # Reference program codes for consistent diversity
        self.diversity_reference_size: int = getattr(config, "diversity_reference_size", 20)

        # Feature scaling infrastructure
        self.feature_stats: Dict[str, Dict[str, Union[float, float, List[float]]]] = {}
        self.feature_scaling_method: str = "minmax"  # Options: minmax, zscore, percentile

        # Per-dimension bins support
        if hasattr(config, "feature_bins") and isinstance(config.feature_bins, dict):
            self.feature_bins_per_dim = config.feature_bins
        else:
            # Backward compatibility - use same bins for all dimensions
            self.feature_bins_per_dim = {
                dim: self.feature_bins for dim in config.feature_dimensions
            }

        logger.info(f"Initialized program database with {len(self.programs)} programs")

        # Novelty judge setup
        from openevolve.embedding import EmbeddingClient

        self.novelty_llm = config.novelty_llm
        self.embedding_client = (
            EmbeddingClient(config.embedding_model) if config.embedding_model else None
        )
        self.similarity_threshold = config.similarity_threshold

    def add(
        self,
        program: Program,
        iteration: int = None,
        target_island: Optional[int] = None,
        skip_novelty: bool = False,
        protected_ids: Optional[Set[str]] = None,
    ) -> str:
        """
        Add a program to the database

        Args:
            program: Program to add
            iteration: Current iteration (defaults to last_iteration)
            target_island: Specific island to add to (auto-detects parent's island if None)
            skip_novelty: Bypass the novelty/similarity filter (used by inject(), where
                a deliberately-placed program must not be silently dropped as a near-duplicate)
            protected_ids: Extra program ids to shield from the population cull this add
                triggers (used by inject() to protect earlier clones in the same batch)

        Returns:
            Program ID
        """
        # A program the evaluator has marked as one it has already measured is
        # not stored at all: it would occupy a population slot, be sampled as a
        # parent and shown as an exemplar, all on another program's numbers.
        # Checked before anything is mutated -- the novelty rejection below runs
        # after the program is already in `self.programs`, and this one must not.
        if self.reject_metric and (program.metrics or {}).get(self.reject_metric):
            logger.info(f"Program {program.id} marked {self.reject_metric} and won't be added")
            return program.id

        # Store the program
        # If iteration is provided, update the program's iteration_found
        if iteration is not None:
            program.iteration_found = iteration
            # Update last_iteration if needed
            self.last_iteration = max(self.last_iteration, iteration)

        self.programs[program.id] = program

        # Calculate feature coordinates for MAP-Elites
        feature_coords = self._calculate_feature_coords(program)

        # Determine target island
        # If target_island is not specified and program has a parent, inherit parent's island
        if target_island is None and program.parent_id:
            parent = self.programs.get(program.parent_id)
            if parent and "island" in parent.metadata:
                # Child inherits parent's island to maintain island isolation
                island_idx = parent.metadata["island"]
                logger.debug(
                    f"Program {program.id} inheriting island {island_idx} from parent {program.parent_id}"
                )
            else:
                # Parent not found or has no island, use current_island
                island_idx = self.current_island
                if parent:
                    logger.warning(
                        f"Parent {program.parent_id} has no island metadata, using current_island {island_idx}"
                    )
                else:
                    logger.warning(
                        f"Parent {program.parent_id} not found, using current_island {island_idx}"
                    )
        elif target_island is not None:
            # Explicit target island specified (e.g., for migrants)
            island_idx = target_island
        else:
            # No parent and no target specified, use current island
            island_idx = self.current_island

        island_idx = island_idx % len(self.islands)  # Ensure valid island

        # Novelty check before adding
        if not skip_novelty and not self._is_novel(program.id, island_idx):
            logger.debug(
                f"Program {program.id} failed in novelty check and won't be added in the island {island_idx}"
            )
            return program.id  # Do not add non-novel program

        # Add to island-specific feature map (replacing existing if better)
        feature_key = self._feature_coords_to_key(feature_coords, group=self._lane_of(program))
        island_feature_map = self.island_feature_maps[island_idx]
        should_replace = feature_key not in island_feature_map

        if not should_replace:
            # Check if the existing program still exists before comparing
            existing_program_id = island_feature_map[feature_key]
            if existing_program_id not in self.programs:
                # Stale reference, replace it
                should_replace = True
                logger.debug(
                    f"Replacing stale program reference {existing_program_id} in island {island_idx} feature map"
                )
            else:
                # Program exists, compare fitness
                should_replace = self._is_better(program, self.programs[existing_program_id])

        # Track a program that gets displaced from its cell so we can remove it
        # from the population if it ends up orphaned (owning no cell, in no island).
        replaced_program_id = None

        if should_replace:
            # Log significant MAP-Elites events
            coords_dict = {
                self.config.feature_dimensions[i]: feature_coords[i]
                for i in range(len(feature_coords))
            }

            if feature_key not in island_feature_map:
                # New cell occupation in this island
                logger.info(
                    "New MAP-Elites cell occupied in island %d: %s", island_idx, coords_dict
                )
                # Check coverage milestone for this island
                total_possible_cells = self.feature_bins ** len(self.config.feature_dimensions)
                island_coverage = (len(island_feature_map) + 1) / total_possible_cells
                if island_coverage in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    logger.info(
                        "Island %d MAP-Elites coverage reached %.1f%% (%d/%d cells)",
                        island_idx,
                        island_coverage * 100,
                        len(island_feature_map) + 1,
                        total_possible_cells,
                    )
            else:
                # Cell replacement - existing program being replaced in this island
                existing_program_id = island_feature_map[feature_key]
                if existing_program_id in self.programs:
                    existing_program = self.programs[existing_program_id]
                    new_fitness = get_fitness_score(program.metrics, self.config.feature_dimensions)
                    existing_fitness = get_fitness_score(
                        existing_program.metrics, self.config.feature_dimensions
                    )
                    logger.info(
                        "Island %d MAP-Elites cell improved: %s (fitness: %.3f -> %.3f)",
                        island_idx,
                        coords_dict,
                        existing_fitness,
                        new_fitness,
                    )

                    # use MAP-Elites to manage archive
                    if existing_program_id in self.archive:
                        self.archive.discard(existing_program_id)
                        self.archive.add(program.id)

                # Remove replaced program from island set to keep it consistent with feature map
                # This prevents accumulation of stale/replaced programs in the island
                self.islands[island_idx].discard(existing_program_id)
                replaced_program_id = existing_program_id

            island_feature_map[feature_key] = program.id

        # Add to island
        self.islands[island_idx].add(program.id)

        # Track which island this program belongs to
        program.metadata["island"] = island_idx

        # Update archive
        self._update_archive(program)

        # Enforce population size limit BEFORE updating best program tracking
        # This ensures newly added programs aren't immediately removed
        self._enforce_population_limit(exclude_program_id=program.id, protected_ids=protected_ids)

        # Update the absolute best program tracking (after population enforcement)
        self._update_best_program(program)

        # Update island-specific best program tracking
        self._update_island_best_program(program, island_idx)

        # If a program was displaced from its cell by this addition, it may now be
        # orphaned - owning no cell and belonging to no island. Such a program is a
        # "zombie" that consumes a population slot but can never be sampled again, so
        # remove it. This runs after best-program tracking is updated so the newly
        # added (better) program is already recorded as best, ensuring we never drop
        # the current best program here.
        if (
            replaced_program_id is not None
            and replaced_program_id != program.id
            and replaced_program_id != self.best_program_id
        ):
            self._remove_program_if_orphaned(replaced_program_id)

        # Save to disk if configured
        if self.config.db_path:
            self._save_program(program)

        logger.debug(f"Added program {program.id} to island {island_idx}")

        return program.id

    def inject(
        self,
        program: Program,
        islands: List[int],
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Add an externally-created program to specific islands, with persistence
        and full accounting.

        Injection places seed / curriculum programs from outside the evolutionary
        loop. For each island the program is cloned with id
        ``f"{program.id}-i{island}"`` and ``metadata["injected"] = True``; a clone
        whose id already exists is skipped, so re-running an injection is
        idempotent. Every clone is added with the novelty filter bypassed and is
        protected -- along with the other clones in this batch -- from the
        population cull its addition triggers, so a low-scoring seed cannot be
        silently dropped. Anything removed as a consequence (a displaced cell
        owner, an orphan, a population cull) is recorded in the eviction journal
        and returned. The caller's ``program`` is not mutated, and no checkpoint
        is written -- the caller owns save placement.

        Returns a report ``{"added", "skipped", "removed"}``: ids that persist,
        ids skipped as already-present, and the journal entries this call produced.
        """
        clone_ids = [f"{program.id}-i{island}" for island in islands]
        batch = set(clone_ids)
        journal_start = len(self.eviction_journal)

        added: List[str] = []
        skipped: List[str] = []
        for island, clone_id in zip(islands, clone_ids):
            if clone_id in self.programs:
                skipped.append(clone_id)
                continue
            clone = Program(
                id=clone_id,
                code=program.code,
                language=program.language,
                parent_id=program.parent_id,
                metrics=dict(program.metrics),
                iteration_found=iteration if iteration is not None else self.last_iteration,
                # A clone carries the evaluator's artifacts. Without them an
                # injected program is the only kind in the archive with no
                # evaluation record, so anything reading artifacts -- a report,
                # a resumed prompt -- sees nothing for exactly the programs a
                # fresh run starts from.
                artifacts_json=program.artifacts_json,
                artifact_dir=program.artifact_dir,
                metadata={
                    **dict(program.metadata),
                    "island": island % len(self.islands),
                    "injected": True,
                },
            )
            self.add(
                clone,
                iteration=iteration,
                target_island=island,
                skip_novelty=True,
                protected_ids=batch,
            )
            added.append(clone_id)

        removed = self.eviction_journal[journal_start:]
        journaled = {entry["id"] for entry in removed}

        # Post-condition: no silent loss. A clone we added is either still present
        # or accounted for in the journal delta (displaced by a same-cell sibling,
        # already listed under `removed`); anything else is a bug.
        present = []
        for clone_id in added:
            if clone_id in self.programs:
                present.append(clone_id)
            elif clone_id not in journaled:
                raise RuntimeError(f"injected program {clone_id} vanished without a journal entry")

        return {"added": present, "skipped": skipped, "removed": removed}

    def get(self, program_id: str) -> Optional[Program]:
        """
        Get a program by ID

        Args:
            program_id: Program ID

        Returns:
            Program or None if not found
        """
        return self.programs.get(program_id)

    def sample(self, num_inspirations: Optional[int] = None) -> Tuple[Program, List[Program]]:
        """
        Sample a program and inspirations for the next evolution step

        Args:
            num_inspirations: Number of inspiration programs to sample (defaults to 5 for backward compatibility)

        Returns:
            Tuple of (parent_program, inspiration_programs)
        """
        # Select parent program
        parent = self._sample_parent()

        # Select inspirations
        if num_inspirations is None:
            num_inspirations = 5  # Default for backward compatibility
        inspirations = self._sample_inspirations(parent, n=num_inspirations)

        logger.debug(f"Sampled parent {parent.id} and {len(inspirations)} inspirations")
        return parent, inspirations

    def sample_from_island(
        self, island_id: int, num_inspirations: Optional[int] = None
    ) -> Tuple[Program, List[Program]]:
        """
        Sample a program and inspirations from a specific island without modifying current_island

        This method is thread-safe and doesn't modify shared state, avoiding race conditions
        when multiple workers sample from different islands concurrently.

        Uses the same exploration/exploitation/random strategy as sample() to ensure
        consistent behavior between single-process and parallel execution modes.

        Args:
            island_id: The island to sample from
            num_inspirations: Number of inspiration programs to sample (defaults to 5)

        Returns:
            Tuple of (parent_program, inspiration_programs)
        """
        # Ensure valid island ID
        island_id = island_id % len(self.islands)

        # Get programs from the specific island
        island_programs = list(self.islands[island_id])

        if not island_programs:
            # Island is empty, fall back to sampling from all programs
            logger.debug(f"Island {island_id} is empty, sampling from all programs")
            return self.sample(num_inspirations)

        # Use exploration_ratio and exploitation_ratio to decide sampling strategy
        # This matches the logic in _sample_parent() for consistent behavior
        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:
            # EXPLORATION: Sample randomly from island (diverse sampling)
            parent = self._sample_from_island_random(island_id)
            sampling_mode = "exploration"
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            # EXPLOITATION: Sample from archive (elite programs)
            parent = self._sample_from_archive_for_island(island_id)
            sampling_mode = "exploitation"
        else:
            # WEIGHTED: Use fitness-weighted sampling (remaining probability)
            parent = self._sample_from_island_weighted(island_id)
            sampling_mode = "weighted"

        # Select inspirations using the same elite/diversity-aware strategy as sample().
        # Pass the requested island explicitly so parallel workers remain isolated even
        # if an archive fallback returns a parent whose metadata points elsewhere.
        if num_inspirations is None:
            num_inspirations = 5  # Default for backward compatibility

        inspirations = self._sample_inspirations(parent, n=num_inspirations, island_id=island_id)

        logger.debug(
            f"Sampled parent {parent.id} and {len(inspirations)} inspirations from island {island_id} "
            f"(mode: {sampling_mode}, rand_val: {rand_val:.3f})"
        )
        return parent, inspirations

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        """
        Get the best program based on a metric

        Args:
            metric: Metric to use for ranking (uses combined_score or average if None)

        Returns:
            Best program or None if database is empty
        """
        if not self.programs:
            return None

        # If no specific metric and we have a tracked best program, return it
        if metric is None and self.best_program_id:
            if self.best_program_id in self.programs:
                logger.debug(f"Using tracked best program: {self.best_program_id}")
                return self.programs[self.best_program_id]
            else:
                logger.warning(
                    f"Tracked best program {self.best_program_id} no longer exists, will recalculate"
                )
                self.best_program_id = None

        # An infeasible program is never reported as best while a feasible one
        # exists; with no constraint configured this is every program.
        pool = breedable(list(self.programs.values()), self.feasibility_metric)

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in pool if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by metric '{metric}': {sorted_programs[0].id}")
        else:
            # Sort by fitness (excluding feature dimensions)
            sorted_programs = sorted(
                pool,
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by fitness score: {sorted_programs[0].id}")

        # Update the best program tracking if we found a better program
        if sorted_programs and (
            self.best_program_id is None or sorted_programs[0].id != self.best_program_id
        ):
            old_id = self.best_program_id
            self.best_program_id = sorted_programs[0].id
            logger.info(f"Updated best program tracking from {old_id} to {self.best_program_id}")

            # Also log the scores to help understand the update
            if (
                old_id
                and old_id in self.programs
                and "combined_score" in self.programs[old_id].metrics
                and "combined_score" in self.programs[self.best_program_id].metrics
            ):
                old_score = self.programs[old_id].metrics["combined_score"]
                new_score = self.programs[self.best_program_id].metrics["combined_score"]
                logger.info(
                    f"Score change: {old_score:.4f} → {new_score:.4f} ({new_score-old_score:+.4f})"
                )

        return sorted_programs[0] if sorted_programs else None

    def get_top_programs(
        self, n: int = 10, metric: Optional[str] = None, island_idx: Optional[int] = None
    ) -> List[Program]:
        """
        Get the top N programs based on a metric

        Args:
            n: Number of programs to return
            metric: Metric to use for ranking (uses average if None)
            island_idx: If specified, only return programs from this island

        Returns:
            List of top programs
        """
        # Validate island_idx parameter
        if island_idx is not None and (island_idx < 0 or island_idx >= len(self.islands)):
            raise IndexError(f"Island index {island_idx} is out of range (0-{len(self.islands)-1})")

        if not self.programs:
            return []

        # Get candidate programs
        if island_idx is not None:
            # Island-specific query
            island_programs = [
                self.programs[pid] for pid in self.islands[island_idx] if pid in self.programs
            ]
            candidates = island_programs
        else:
            # Global query
            candidates = list(self.programs.values())

        if not candidates:
            return []

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in candidates if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
        else:
            # Sort by combined_score if available, otherwise by average of all numeric metrics,
            # with any infeasible candidate placed after every feasible one (rank_exemplars),
            # and one entry per distinct behaviour where the evaluator names one.
            sorted_programs = dedup_exemplars(self.exemplar_order(candidates), self.dedup_key)

        return sorted_programs[:n]

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # Perform artifact cleanup before saving
        self._cleanup_old_artifacts(save_path)

        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in self.programs.values():
            prompts = None
            if (
                self.config.log_prompts
                and self.prompts_by_program
                and program.id in self.prompts_by_program
            ):
                prompts = self.prompts_by_program[program.id]
            self._save_program(program, save_path, prompts=prompts)

        # Save metadata
        metadata = {
            "island_feature_maps": self.island_feature_maps,
            "islands": [list(island) for island in self.islands],
            "archive": list(self.archive),
            "best_program_id": self.best_program_id,
            "island_best_programs": self.island_best_programs,
            "last_iteration": iteration or self.last_iteration,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
            "feature_stats": self._serialize_feature_stats(),
            "eviction_journal": self.eviction_journal,
        }

        # Per-group summary, readable straight from the checkpoint without a load.
        # Derivable, so load() does not restore it. Group-int keys become strings
        # through json, as with any int-keyed metadata field.
        lane_report = self.lane_report()
        if lane_report:
            metadata["lane_report"] = lane_report

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved database with {len(self.programs)} programs to {save_path}")

    def load(self, path: str) -> None:
        """
        Load the database from disk

        Args:
            path: Path to load from
        """
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return

        # Load metadata first
        metadata_path = os.path.join(path, "metadata.json")
        saved_islands = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.island_feature_maps = metadata.get(
                "island_feature_maps", [{} for _ in range(self.config.num_islands)]
            )
            # lane_split_grids expects group-prefixed cell keys ("g:coords"). A
            # checkpoint saved under the shared grid has bare keys; resuming it
            # under split grids would mix key formats and corrupt cell ownership.
            # Warn loudly rather than silently degrade -- the fix is to re-key the
            # checkpoint before resuming.
            if self.lane_split_grids and any(
                keys and not all(":" in k for k in keys)
                for keys in (m.keys() for m in self.island_feature_maps)
            ):
                logger.warning(
                    "lane_split_grids is on but %s has un-prefixed cell keys -- this "
                    "checkpoint predates per-group grids and must be re-keyed before "
                    "resuming, or cell ownership will corrupt.",
                    path,
                )
            saved_islands = metadata.get("islands", [])
            self.archive = set(metadata.get("archive", []))
            self.best_program_id = metadata.get("best_program_id")
            self.island_best_programs = metadata.get(
                "island_best_programs", [None] * len(saved_islands)
            )
            self.last_iteration = metadata.get("last_iteration", 0)
            self.current_island = metadata.get("current_island", 0)
            self.island_generations = metadata.get("island_generations", [0] * len(saved_islands))
            self.last_migration_generation = metadata.get("last_migration_generation", 0)

            # Load feature_stats for MAP-Elites grid stability
            self.feature_stats = self._deserialize_feature_stats(metadata.get("feature_stats", {}))
            self.eviction_journal = metadata.get("eviction_journal", [])

            logger.info(f"Loaded database metadata with last_iteration={self.last_iteration}")
            if self.feature_stats:
                logger.info(f"Loaded feature_stats for {len(self.feature_stats)} dimensions")

        # Load programs
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        program = Program.from_dict(program_data)
                        self.programs[program.id] = program
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        # Reconstruct island assignments from metadata
        self._reconstruct_islands(saved_islands)

        # Ensure island_generations list has correct length
        if len(self.island_generations) != len(self.islands):
            self.island_generations = [0] * len(self.islands)

        # Ensure island_best_programs list has correct length
        if len(self.island_best_programs) != len(self.islands):
            self.island_best_programs = [None] * len(self.islands)

        logger.info(f"Loaded database with {len(self.programs)} programs from {path}")

        # Log the reconstructed island status
        self.log_island_status()

    def _reconstruct_islands(self, saved_islands: List[List[str]]) -> None:
        """
        Reconstruct island assignments from saved metadata

        Args:
            saved_islands: List of island program ID lists from metadata
        """
        # Initialize empty islands
        num_islands = max(len(saved_islands), self.config.num_islands)
        self.islands = [set() for _ in range(num_islands)]

        missing_programs = []
        restored_programs = 0

        # Restore island assignments
        for island_idx, program_ids in enumerate(saved_islands):
            if island_idx >= len(self.islands):
                continue

            for program_id in program_ids:
                if program_id in self.programs:
                    # Program exists, add to island
                    self.islands[island_idx].add(program_id)
                    # Set island metadata on the program
                    self.programs[program_id].metadata["island"] = island_idx
                    restored_programs += 1
                else:
                    # Program missing, track it
                    missing_programs.append((island_idx, program_id))

        # Clean up archive - remove missing programs
        original_archive_size = len(self.archive)
        self.archive = {pid for pid in self.archive if pid in self.programs}

        # Clean up island_feature_maps - remove missing programs
        feature_keys_to_remove = []
        for island_idx, island_map in enumerate(self.island_feature_maps):
            island_keys_to_remove = []
            for key, program_id in island_map.items():
                if program_id not in self.programs:
                    island_keys_to_remove.append(key)
                    feature_keys_to_remove.append((island_idx, key))
            for key in island_keys_to_remove:
                del island_map[key]

        # Clean up island best programs - remove stale references
        self._cleanup_stale_island_bests()

        # Check best program
        if self.best_program_id and self.best_program_id not in self.programs:
            logger.warning(f"Best program {self.best_program_id} not found, will recalculate")
            self.best_program_id = None

        # Log reconstruction results
        if missing_programs:
            logger.warning(
                f"Found {len(missing_programs)} missing programs during island reconstruction:"
            )
            for island_idx, program_id in missing_programs[:5]:  # Show first 5
                logger.warning(f"  Island {island_idx}: {program_id}")
            if len(missing_programs) > 5:
                logger.warning(f"  ... and {len(missing_programs) - 5} more")

        if original_archive_size > len(self.archive):
            logger.info(
                f"Removed {original_archive_size - len(self.archive)} missing programs from archive"
            )

        if feature_keys_to_remove:
            logger.info(
                f"Removed {len(feature_keys_to_remove)} missing programs from island feature maps"
            )

        logger.info(f"Reconstructed islands: restored {restored_programs} programs to islands")

        # If we have programs but no island assignments, distribute them
        if self.programs and sum(len(island) for island in self.islands) == 0:
            logger.info("No island assignments found, distributing programs across islands")
            self._distribute_programs_to_islands()

    def _distribute_programs_to_islands(self) -> None:
        """
        Distribute loaded programs across islands when no island metadata exists
        """
        program_ids = list(self.programs.keys())

        # Distribute programs round-robin across islands
        for i, program_id in enumerate(program_ids):
            island_idx = i % len(self.islands)
            self.islands[island_idx].add(program_id)
            self.programs[program_id].metadata["island"] = island_idx

        logger.info(f"Distributed {len(program_ids)} programs across {len(self.islands)} islands")

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
            prompts: Optional prompts to save with the program, in the format {template_key: { 'system': str, 'user': str }}
        """
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_dict = program.to_dict()
        if prompts:
            program_dict["prompts"] = prompts
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f)

    def _calculate_feature_coords(self, program: Program) -> List[int]:
        """
        Calculate feature coordinates for the MAP-Elites grid

        Args:
            program: Program to calculate features for

        Returns:
            List of feature coordinates
        """
        coords = []

        for dim in self.config.feature_dimensions:
            # PRIORITY 1: Check if this is a custom metric from the evaluator
            # This allows users to override built-in features with their own implementations
            if dim in program.metrics:
                # Use custom metric from evaluator
                score = program.metrics[dim]
                # Update stats and scale
                self._update_feature_stats(dim, score)
                scaled_value = self._scale_feature_value(dim, score)
                num_bins = self.feature_bins_per_dim.get(dim, self.feature_bins)
                bin_idx = int(scaled_value * num_bins)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            # PRIORITY 2: Fall back to built-in features if not in metrics
            elif dim == "complexity":
                # Use code length as complexity measure
                complexity = len(program.code)
                bin_idx = self._calculate_complexity_bin(complexity)
                coords.append(bin_idx)
            elif dim == "diversity":
                # Use cached diversity calculation with reference set
                if len(self.programs) < 2:
                    bin_idx = 0
                else:
                    diversity = self._get_cached_diversity(program)
                    bin_idx = self._calculate_diversity_bin(diversity)
                coords.append(bin_idx)
            elif dim == "score":
                # Use average of numeric metrics
                if not program.metrics:
                    bin_idx = 0
                else:
                    # Use fitness score for "score" dimension (consistent with rest of system)
                    avg_score = get_fitness_score(program.metrics, self.config.feature_dimensions)
                    # Update stats and scale
                    self._update_feature_stats("score", avg_score)
                    scaled_value = self._scale_feature_value("score", avg_score)
                    num_bins = self.feature_bins_per_dim.get("score", self.feature_bins)
                    bin_idx = int(scaled_value * num_bins)
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            elif getattr(self.config, "missing_feature_policy", "error") == "floor":
                # Bin at the axis floor rather than losing the whole iteration.
                # The program is still placed; a floored coordinate is honest
                # about carrying no information on this axis.
                logger.warning(
                    "Program %s lacks feature dimension '%s'; binning at the axis "
                    "floor (missing_feature_policy=floor). Available metrics: %s",
                    program.id,
                    dim,
                    list(program.metrics.keys()),
                )
                coords.append(0)
            else:
                # Feature not found - this is an error
                raise ValueError(
                    f"Feature dimension '{dim}' specified in config but not found in program metrics. "
                    f"Available metrics: {list(program.metrics.keys())}. "
                    f"Built-in features: 'complexity', 'diversity', 'score'. "
                    f"Either remove '{dim}' from feature_dimensions or ensure your evaluator returns it."
                )
        # Only log coordinates at debug level for troubleshooting
        logger.debug(
            "MAP-Elites coords: %s",
            str({self.config.feature_dimensions[i]: coords[i] for i in range(len(coords))}),
        )
        return coords

    def _calculate_complexity_bin(self, complexity: int) -> int:
        """
        Calculate the bin index for a given complexity value using feature scaling.

        Args:
            complexity: The complexity value (code length)

        Returns:
            Bin index in range [0, self.feature_bins - 1]
        """
        # Update feature statistics
        self._update_feature_stats("complexity", float(complexity))

        # Scale the value using configured method
        scaled_value = self._scale_feature_value("complexity", float(complexity))

        # Get number of bins for this dimension
        num_bins = self.feature_bins_per_dim.get("complexity", self.feature_bins)

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _calculate_diversity_bin(self, diversity: float) -> int:
        """
        Calculate the bin index for a given diversity value using feature scaling.

        Args:
            diversity: The average fast code diversity to other programs

        Returns:
            Bin index in range [0, self.feature_bins - 1]
        """
        # Update feature statistics
        self._update_feature_stats("diversity", diversity)

        # Scale the value using configured method
        scaled_value = self._scale_feature_value("diversity", diversity)

        # Get number of bins for this dimension
        num_bins = self.feature_bins_per_dim.get("diversity", self.feature_bins)

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _feature_coords_to_key(self, coords: List[int], group: Any = None) -> str:
        """
        Convert feature coordinates to a string key

        Args:
            coords: Feature coordinates
            group: Lane group to prefix (only used when lane_split_grids is on);
                gives each group its own grid so keys never collide across groups

        Returns:
            String key
        """
        key = "-".join(str(c) for c in coords)
        if self.lane_split_grids and group is not None:
            return f"{group}:{key}"
        return key

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Adapted from SakanaAI/ShinkaEvolve (Apache-2.0 License)
        Original source: https://github.com/SakanaAI/ShinkaEvolve/blob/main/shinka/database/dbase.py#L1452

        Compute cosine similarity between two vectors.
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        arr1 = np.array(vec1, dtype=np.float32)
        arr2 = np.array(vec2, dtype=np.float32)

        norm_a = np.linalg.norm(arr1)
        norm_b = np.linalg.norm(arr2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(arr1, arr2) / (norm_a * norm_b)

        return float(similarity)

    def _llm_judge_novelty(self, program: Program, similar_program: Program) -> bool:
        """
        Use LLM to judge if a program is novel compared to a similar existing program
        """
        import asyncio

        from openevolve.novelty_judge import NOVELTY_SYSTEM_MSG, NOVELTY_USER_MSG

        user_msg = NOVELTY_USER_MSG.format(
            language=program.language,
            existing_code=similar_program.code,
            proposed_code=program.code,
        )

        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to run in a new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.novelty_llm.generate_with_context(
                            system_message=NOVELTY_SYSTEM_MSG,
                            messages=[{"role": "user", "content": user_msg}],
                        ),
                    )
                    content: str = future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                content: str = asyncio.run(
                    self.novelty_llm.generate_with_context(
                        system_message=NOVELTY_SYSTEM_MSG,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                )

            if content is None or content is None:
                logger.warning("Novelty LLM returned empty response")
                return True

            content = content.strip()

            # Parse the response
            NOVEL_i = content.upper().find("NOVEL")
            NOT_NOVEL_i = content.upper().find("NOT NOVEL")

            if NOVEL_i == -1 and NOT_NOVEL_i == -1:
                logger.warning(f"Unexpected novelty LLM response: {content}")
                return True  # Assume novel if we can't parse

            if NOVEL_i != -1 and NOT_NOVEL_i != -1:
                # Both found, take the one that appears first
                is_novel = NOVEL_i < NOT_NOVEL_i
            elif NOVEL_i != -1:
                is_novel = True
            else:
                is_novel = False

            return is_novel

        except Exception as e:
            logger.error(f"Error in novelty LLM check: {e}")

        return True

    def _is_novel(self, program_id: int, island_idx: int) -> bool:
        """
        Determine if a program is novel based on diversity to existing programs

        Args:
            program: Program to check
            island_idx: Island index

        Returns:
            True if novel, False otherwise
        """
        if self.embedding_client is None or self.similarity_threshold <= 0.0:
            # Novelty checking disabled
            return True

        program = self.programs[program_id]
        embd = self.embedding_client.get_embedding(program.code)
        self.programs[program_id].embedding = embd

        max_smlty = float("-inf")
        max_smlty_pid = None

        for pid in self.islands[island_idx]:
            other = self.programs[pid]

            if other.embedding is None:
                logger.warning(f"Program {other.id} has no embedding, skipping similarity check")
                continue

            similarity = self._cosine_similarity(embd, other.embedding)

            if similarity >= max(max_smlty, self.similarity_threshold):
                max_smlty = similarity
                max_smlty_pid = pid

        if max_smlty_pid is None:
            # No similar programs found, consider it novel
            return True

        return self._llm_judge_novelty(program, self.programs[max_smlty_pid])

    def exemplar_order(self, programs: list, score=None) -> list:
        """`rank_exemplars` with this database's constraint settings."""
        return rank_exemplars(
            programs,
            self.feasibility_metric,
            self.violation_metric,
            self.config.feature_dimensions,
            score,
        )

    def _breedable_ids(self, ids: list) -> list:
        """`ids` narrowed to what may be bred from.

        The feasible ones when the scope has any, topped up to
        `feasibility_min_pool` with the closest infeasible ones by violation so
        a narrow miss stays reachable. When it has none — a population that has
        not yet produced a feasible program — the closest half by violation,
        which keeps a gradient without collapsing selection onto a single
        program and losing the diversity the search needs. The identity when
        unconfigured.

        With `lane_metric` set the rule is applied to each group separately and
        the results unioned, because a group absent from this pool cannot be
        reached by any sampler downstream: the group-balanced draw
        (`lane_sampling_gamma`) can only balance groups that are in it, so one
        group holding every feasible member would otherwise silence the rest.
        """
        if self.feasibility_metric is None or not ids:
            return ids
        known = [pid for pid in ids if pid in self.programs]
        if not known:
            return ids
        if self.lane_metric is None:
            return self._breedable_within(known, ids)
        groups: Dict[Any, list] = {}
        for pid in known:
            groups.setdefault(self._lane_of(self.programs[pid]), []).append(pid)
        kept = set()
        for members in groups.values():
            kept.update(self._breedable_within(members, members))
        return [pid for pid in known if pid in kept]

    def _breedable_within(self, known: list, fallback: list) -> list:
        """The breedable rule over one scope: feasible topped up to the floor,
        else the closest half by violation. `fallback` is returned unchanged
        when no violation metric orders an all-infeasible scope."""
        feasible = [
            pid for pid in known if is_feasible(self.programs[pid], self.feasibility_metric)
        ]
        if feasible:
            return self._topped_up(feasible, known)
        if self.violation_metric is None:
            return fallback
        return self._closest_by_violation(known)[: max(1, len(known) // 2)]

    def _topped_up(self, feasible: list, known: list) -> list:
        """`feasible` extended to `feasibility_min_pool` with the closest
        infeasible members of `known`. Identity when the floor is already met or
        no violation metric orders the candidates."""
        shortfall = self.feasibility_min_pool - len(feasible)
        if shortfall <= 0 or self.violation_metric is None:
            return feasible
        chosen = set(feasible)
        nearest = [pid for pid in self._closest_by_violation(known) if pid not in chosen]
        return feasible + nearest[:shortfall]

    def _closest_by_violation(self, ids: list) -> list:
        """`ids` ordered by ascending violation; an absent measurement is worst."""
        return sorted(ids, key=lambda pid: violation_of(self.programs[pid], self.violation_metric))

    def _is_better(self, program1: Program, program2: Program) -> bool:
        """
        Determine if program1 has better FITNESS than program2

        Uses fitness calculation that excludes MAP-Elites feature dimensions
        to prevent pollution of fitness comparisons.

        Args:
            program1: First program
            program2: Second program

        Returns:
            True if program1 is better than program2
        """
        # Feasibility outranks fitness: an infeasible program is kept for its
        # lineage but never preferred. Two infeasible programs compare by how
        # far from feasible they are, which is what gives a wholly infeasible
        # population a direction to move in. Inert unless configured.
        if self.feasibility_metric is not None:
            f1 = is_feasible(program1, self.feasibility_metric)
            f2 = is_feasible(program2, self.feasibility_metric)
            if f1 != f2:
                return f1
            if not f1:
                v1 = violation_of(program1, self.violation_metric)
                v2 = violation_of(program2, self.violation_metric)
                if v1 != v2:
                    return v1 < v2

        # If no metrics, use newest
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        # If only one has metrics, it's better
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        # Compare fitness (excluding feature dimensions)
        fitness1 = get_fitness_score(program1.metrics, self.config.feature_dimensions)
        fitness2 = get_fitness_score(program2.metrics, self.config.feature_dimensions)

        # A coarse fitness leaves programs exactly tied, and which of two equals
        # is kept is then an accident of arrival order. `tiebreak_metric` makes
        # it a preference instead. Inert unless configured.
        if fitness1 == fitness2 and self.tiebreak_metric is not None:
            return self._wins_tiebreak(program1, program2)

        return fitness1 > fitness2

    def _wins_tiebreak(self, program1: Program, program2: Program) -> bool:
        """Whether `program1` wins on `tiebreak_metric`. A program carrying no
        measurement loses to one that does, and two unmeasured or two equal
        programs fall through to upstream's strict `>`, i.e. False."""
        t1 = tiebreak_of(program1, self.tiebreak_metric)
        t2 = tiebreak_of(program2, self.tiebreak_metric)
        if t1 is None or t2 is None:
            return t2 is None and t1 is not None
        return t1 < t2 if self.tiebreak_lower_is_better else t1 > t2

    def _update_archive(self, program: Program) -> None:
        """
        Update the archive of elite programs

        Args:
            program: Program to consider for archive
        """
        # If archive not full, add program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Clean up stale references and get valid archive programs
        valid_archive_programs = []
        stale_ids = []

        for pid in self.archive:
            if pid in self.programs:
                valid_archive_programs.append(self.programs[pid])
            else:
                stale_ids.append(pid)

        # Remove stale references from archive
        for stale_id in stale_ids:
            self.archive.discard(stale_id)
            logger.debug(f"Removing stale program {stale_id} from archive")

        # If archive is now not full after cleanup, just add the new program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Find worst program among valid programs
        if valid_archive_programs:
            worst_program = min(
                valid_archive_programs,
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
            )

            # Replace if new program is better
            if self._is_better(program, worst_program):
                self.archive.remove(worst_program.id)
                self.archive.add(program.id)
        else:
            # No valid programs in archive, just add the new one
            self.archive.add(program.id)

    def _update_best_program(self, program: Program) -> None:
        """
        Update the absolute best program tracking

        Args:
            program: Program to consider as the new best
        """
        # If we don't have a best program yet, this becomes the best
        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.debug(f"Set initial best program to {program.id}")
            return

        # Compare with current best program (if it still exists)
        if self.best_program_id not in self.programs:
            logger.warning(
                f"Best program {self.best_program_id} no longer exists, clearing reference"
            )
            self.best_program_id = program.id
            logger.info(f"Set new best program to {program.id}")
            return

        current_best = self.programs[self.best_program_id]

        # Update if the new program is better
        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id

            # Log the change
            if "combined_score" in program.metrics and "combined_score" in current_best.metrics:
                old_score = current_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.info(
                    f"New best program {program.id} replaces {old_id} (combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.info(f"New best program {program.id} replaces {old_id}")

    def _update_island_best_program(self, program: Program, island_idx: int) -> None:
        """
        Update the best program tracking for a specific island

        Args:
            program: Program to consider as the new best for the island
            island_idx: Island index
        """
        # Ensure island_idx is valid
        if island_idx >= len(self.island_best_programs):
            logger.warning(f"Invalid island index {island_idx}, skipping island best update")
            return

        # If island doesn't have a best program yet, this becomes the best
        current_island_best_id = self.island_best_programs[island_idx]
        if current_island_best_id is None:
            self.island_best_programs[island_idx] = program.id
            logger.debug(f"Set initial best program for island {island_idx} to {program.id}")
            return

        # Check if current best still exists
        if current_island_best_id not in self.programs:
            logger.warning(
                f"Island {island_idx} best program {current_island_best_id} no longer exists, updating to {program.id}"
            )
            self.island_best_programs[island_idx] = program.id
            return

        current_island_best = self.programs[current_island_best_id]

        # Update if the new program is better
        if self._is_better(program, current_island_best):
            old_id = current_island_best_id
            self.island_best_programs[island_idx] = program.id

            # Log the change
            if (
                "combined_score" in program.metrics
                and "combined_score" in current_island_best.metrics
            ):
                old_score = current_island_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id} "
                    f"(combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id}"
                )

    def _sample_parent(self) -> Program:
        """
        Sample a parent program from the current island for the next evolution step

        Returns:
            Parent program from current island
        """
        # Use exploration_ratio and exploitation_ratio to decide sampling strategy
        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:
            # EXPLORATION: Sample from current island (diverse sampling)
            return self._sample_exploration_parent()
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            # EXPLOITATION: Sample from archive (elite programs)
            return self._sample_exploitation_parent()
        else:
            # RANDOM: Sample from any program (remaining probability)
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        """
        Sample a parent for exploration (from current island)
        """
        current_island_programs = self.islands[self.current_island]

        if not current_island_programs:
            # If current island is empty, initialize with best program or random program
            if self.best_program_id and self.best_program_id in self.programs:
                # Create a copy of best program for the empty island (don't reuse same ID)
                best_program = self.programs[self.best_program_id]
                copy_program = Program(
                    id=str(uuid.uuid4()),
                    code=best_program.code,
                    changes_description=best_program.changes_description,
                    language=best_program.language,
                    parent_id=best_program.id,
                    generation=best_program.generation,
                    timestamp=time.time(),
                    iteration_found=self.last_iteration,
                    metrics=best_program.metrics.copy(),
                    complexity=best_program.complexity,
                    diversity=best_program.diversity,
                    metadata={"island": self.current_island},
                    artifacts_json=best_program.artifacts_json,
                    artifact_dir=best_program.artifact_dir,
                )
                self.programs[copy_program.id] = copy_program
                self.islands[self.current_island].add(copy_program.id)
                logger.debug(
                    f"Initialized empty island {self.current_island} with copy of best program"
                )
                return copy_program
            else:
                # Use any available program
                return next(iter(self.programs.values()))

        # Clean up stale references and sample from current island. Stale means
        # absent from `programs`; a program the breedable filter declined is
        # still a member of the island and must survive the cleanup.
        live_programs = [pid for pid in current_island_programs if pid in self.programs]

        # Remove stale program IDs from island
        if len(live_programs) < len(current_island_programs):
            stale_ids = current_island_programs - set(live_programs)
            logger.debug(
                f"Removing {len(stale_ids)} stale program IDs from island {self.current_island}"
            )
            for stale_id in stale_ids:
                self.islands[self.current_island].discard(stale_id)

        valid_programs = self._breedable_ids(live_programs)

        # If no valid programs after cleanup, reinitialize island
        if not valid_programs:
            logger.warning(
                f"Island {self.current_island} has no valid programs after cleanup, reinitializing"
            )
            if self.best_program_id and self.best_program_id in self.programs:
                # Create a copy of best program for the empty island (don't reuse same ID)
                best_program = self.programs[self.best_program_id]
                copy_program = Program(
                    id=str(uuid.uuid4()),
                    code=best_program.code,
                    changes_description=best_program.changes_description,
                    language=best_program.language,
                    parent_id=best_program.id,
                    generation=best_program.generation,
                    timestamp=time.time(),
                    iteration_found=self.last_iteration,
                    metrics=best_program.metrics.copy(),
                    complexity=best_program.complexity,
                    diversity=best_program.diversity,
                    metadata={"island": self.current_island},
                    artifacts_json=best_program.artifacts_json,
                    artifact_dir=best_program.artifact_dir,
                )
                self.programs[copy_program.id] = copy_program
                self.islands[self.current_island].add(copy_program.id)
                logger.debug(
                    f"Reinitialized empty island {self.current_island} with copy of best program"
                )
                return copy_program
            else:
                return next(iter(self.programs.values()))

        # Group-balanced selection when configured, else uniform (upstream).
        if self.lane_sampling_gamma is not None:
            return self._group_weighted_choice(valid_programs)
        parent_id = random.choice(valid_programs)
        return self.programs[parent_id]

    def _sample_exploitation_parent(self) -> Program:
        """
        Sample a parent for exploitation (from archive/elite programs)
        """
        if not self.archive:
            # Fallback to exploration if no archive
            return self._sample_exploration_parent()

        # Clean up stale references in archive. As on the island path, only a
        # program absent from `programs` is stale -- an infeasible one keeps its
        # archive membership so the violation fallback can still reach it.
        live_archive = [pid for pid in self.archive if pid in self.programs]

        # Remove stale program IDs from archive
        if len(live_archive) < len(self.archive):
            stale_ids = self.archive - set(live_archive)
            logger.debug(f"Removing {len(stale_ids)} stale program IDs from archive")
            for stale_id in stale_ids:
                self.archive.discard(stale_id)

        valid_archive = self._breedable_ids(live_archive)

        # If no valid archive programs, fallback to exploration
        if not valid_archive:
            logger.warning(
                "Archive has no valid programs after cleanup, falling back to exploration"
            )
            return self._sample_exploration_parent()

        # Prefer programs from current island in archive
        archive_programs_in_island = [
            pid
            for pid in valid_archive
            if self.programs[pid].metadata.get("island") == self.current_island
        ]

        if archive_programs_in_island:
            parent_id = random.choice(self._breedable_ids(archive_programs_in_island))
            return self.programs[parent_id]
        else:
            # Fall back to any valid archive program if current island has none
            parent_id = random.choice(self._breedable_ids(valid_archive))
            return self.programs[parent_id]

    def _sample_random_parent(self) -> Program:
        """
        Sample a completely random parent from all programs
        """
        if not self.programs:
            raise ValueError("No programs available for sampling")

        # Sample randomly from all programs
        program_id = random.choice(self._breedable_ids(list(self.programs.keys())))
        return self.programs[program_id]

    def _sample_from_island_weighted(self, island_id: int) -> Program:
        """
        Sample a parent from a specific island using fitness-weighted selection

        Args:
            island_id: The island to sample from

        Returns:
            Parent program selected using fitness-weighted sampling
        """
        island_id = island_id % len(self.islands)
        island_programs = self._breedable_ids(list(self.islands[island_id]))

        if not island_programs:
            # Island is empty, fall back to any available program
            logger.debug(f"Island {island_id} is empty, sampling from all programs")
            return self._sample_random_parent()

        # Select parent from island programs
        if len(island_programs) == 1:
            parent_id = island_programs[0]
        else:
            # Use weighted sampling based on program scores
            island_program_objects = [
                self.programs[pid] for pid in island_programs if pid in self.programs
            ]

            if not island_program_objects:
                # Fallback if programs not found
                parent_id = random.choice(island_programs)
            else:
                # Calculate weights based on fitness scores
                weights = []
                for prog in island_program_objects:
                    fitness = get_fitness_score(prog.metrics, self.config.feature_dimensions)
                    # Add small epsilon to avoid zero weights
                    weights.append(max(fitness, 0.001))

                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [1.0 / len(island_program_objects)] * len(island_program_objects)

                # Sample parent based on weights
                parent = random.choices(island_program_objects, weights=weights, k=1)[0]
                parent_id = parent.id

        parent = self.programs.get(parent_id)
        if not parent:
            # Should not happen, but handle gracefully
            logger.error(f"Parent program {parent_id} not found in database")
            return self._sample_random_parent()

        return parent

    def _sample_from_island_random(self, island_id: int) -> Program:
        """
        Sample a completely random parent from a specific island (uniform distribution)

        Args:
            island_id: The island to sample from

        Returns:
            Parent program selected uniformly at random
        """
        island_id = island_id % len(self.islands)
        island_programs = self._breedable_ids(list(self.islands[island_id]))

        if not island_programs:
            # Island is empty, fall back to any available program
            logger.debug(f"Island {island_id} is empty, sampling from all programs")
            return self._sample_random_parent()

        # Clean up stale references
        valid_programs = self._breedable_ids(
            [pid for pid in island_programs if pid in self.programs]
        )

        if not valid_programs:
            logger.warning(
                f"Island {island_id} has no valid programs, falling back to random sampling"
            )
            return self._sample_random_parent()

        # Group-balanced selection when configured, else uniform (upstream).
        if self.lane_sampling_gamma is not None:
            return self._group_weighted_choice(valid_programs)
        parent_id = random.choice(valid_programs)
        return self.programs[parent_id]

    def _sample_from_archive_for_island(self, island_id: int) -> Program:
        """
        Sample a parent from the archive, preferring programs from the specified island

        Args:
            island_id: The island to prefer programs from

        Returns:
            Parent program from archive (preferably from the specified island)
        """
        if not self.archive:
            # Fallback to weighted sampling from island
            logger.debug(f"Archive is empty, falling back to weighted island sampling")
            return self._sample_from_island_weighted(island_id)

        # Clean up stale references in archive
        valid_archive = self._breedable_ids([pid for pid in self.archive if pid in self.programs])

        if not valid_archive:
            logger.warning(
                "Archive has no valid programs, falling back to weighted island sampling"
            )
            return self._sample_from_island_weighted(island_id)

        island_id = island_id % len(self.islands)

        # Prefer programs from the specified island in archive
        archive_programs_in_island = [
            pid for pid in valid_archive if self.programs[pid].metadata.get("island") == island_id
        ]

        if archive_programs_in_island:
            parent_id = random.choice(archive_programs_in_island)
            return self.programs[parent_id]
        else:
            # Fall back to any valid archive program if island has none
            parent_id = random.choice(valid_archive)
            return self.programs[parent_id]

    def _sample_inspirations_scoped(
        self, parent: Program, n: int, island_id: Optional[int]
    ) -> Optional[List[Program]]:
        """Group-scoped inspirations: the parent's own group best, then a few
        honest cross-group exemplars filling the tail.

        Fills the first ``n - lane_cross_inspirations`` slots (at least one) with
        the highest-scoring same-group island members, then up to
        ``lane_cross_inspirations`` best members of other groups -- so the prompt
        shows a group its own exemplars while keeping a labelled cross-group
        reference. Returns None when the parent's group has no other island
        members, so the caller falls back to the upstream (group-blind) sampler.
        """
        if island_id is None:
            parent_island = parent.metadata.get("island", self.current_island)
        else:
            parent_island = island_id
        parent_island %= len(self.islands)

        parent_group = self._lane_of(parent)
        members = [
            self.programs[pid]
            for pid in self.islands[parent_island]
            if pid in self.programs and pid != parent.id
        ]
        same = [p for p in members if self._lane_of(p) == parent_group]
        if not same:  # thin/empty group on this island: upstream owns the fallback
            return None

        same = dedup_exemplars(self.exemplar_order(same, score=_group_score), self.dedup_key)
        k = max(0, self.lane_cross_inspirations)
        chosen = same[: max(1, n - k)]
        if k and len(chosen) < n:
            cross = [p for p in members if self._lane_of(p) != parent_group]
            cross = dedup_exemplars(self.exemplar_order(cross, score=_group_score), self.dedup_key)
            chosen = chosen + cross[: min(k, n - len(chosen))]
        return chosen[:n]

    def _sample_inspirations(
        self, parent: Program, n: int = 5, island_id: Optional[int] = None
    ) -> List[Program]:
        """
        Sample inspiration programs for the next evolution step.

        For proper island-based evolution, inspirations are sampled ONLY from the
        current island, maintaining genetic isolation between islands.

        Args:
            parent: Parent program
            n: Number of inspirations to sample
            island_id: Explicit island to sample from. If omitted, use the
                parent program's island metadata.

        Returns:
            List of inspiration programs from the current island
        """
        # Group-scoped composition when configured: same-group best plus a few
        # honestly-labelled cross-group exemplars (see _sample_inspirations_scoped).
        if self.lane_prompt_scope:
            scoped = self._sample_inspirations_scoped(parent, n, island_id)
            if scoped is not None:
                return scoped

        inspirations = []

        # Prefer an explicitly requested island. This matters for
        # sample_from_island(), where archive fallback may return a parent whose
        # metadata belongs to a different island.
        if island_id is None:
            parent_island = parent.metadata.get("island", self.current_island)
        else:
            parent_island = island_id

        parent_island %= len(self.islands)

        # Get all programs from the current island
        island_program_ids = list(self.islands[parent_island])
        island_programs = [self.programs[pid] for pid in island_program_ids if pid in self.programs]

        if not island_programs:
            logger.warning(f"Island {parent_island} has no programs for inspiration sampling")
            return []

        # Include the island's best program if available and different from parent
        island_best_id = self.island_best_programs[parent_island]
        if (
            island_best_id is not None
            and island_best_id != parent.id
            and island_best_id in self.programs
        ):
            island_best = self.programs[island_best_id]
            inspirations.append(island_best)
            logger.debug(
                f"Including island {parent_island} best program {island_best_id} in inspirations"
            )
        elif island_best_id is not None and island_best_id not in self.programs:
            # Clean up stale island best reference
            logger.warning(
                f"Island {parent_island} best program {island_best_id} no longer exists, clearing reference"
            )
            self.island_best_programs[parent_island] = None

        # Add top programs from the island as inspirations
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_island_programs = self.get_top_programs(n=top_n, island_idx=parent_island)
        for program in top_island_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent.id:
                inspirations.append(program)

        # Add diverse programs from within the island
        if len(island_programs) > n and len(inspirations) < n:
            remaining_slots = n - len(inspirations)

            # Try to sample from different feature cells within the island
            feature_coords = self._calculate_feature_coords(parent)
            nearby_programs = []

            # Create a mapping of feature cells to island programs for efficient lookup
            island_feature_map = {}
            for prog_id in island_program_ids:
                if prog_id in self.programs:
                    prog = self.programs[prog_id]
                    prog_coords = self._calculate_feature_coords(prog)
                    cell_key = self._feature_coords_to_key(prog_coords, group=self._lane_of(prog))
                    island_feature_map[cell_key] = prog_id

            # Try to find programs from nearby feature cells within the island. With
            # split grids the perturbed lookup uses the parent's group prefix, so it
            # searches within the parent's own lane.
            parent_group = self._lane_of(parent)
            for _ in range(remaining_slots * 3):  # Try more times to find nearby programs
                # Perturb coordinates
                perturbed_coords = [
                    max(0, min(self.feature_bins - 1, c + random.randint(-2, 2)))
                    for c in feature_coords
                ]

                cell_key = self._feature_coords_to_key(perturbed_coords, group=parent_group)
                if cell_key in island_feature_map:
                    program_id = island_feature_map[cell_key]
                    if (
                        program_id != parent.id
                        and program_id not in [p.id for p in inspirations]
                        and program_id not in [p.id for p in nearby_programs]
                        and program_id in self.programs
                    ):
                        nearby_programs.append(self.programs[program_id])
                        if len(nearby_programs) >= remaining_slots:
                            break

            # If we still need more, add random programs from the island
            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)

                # Get available programs from the island
                excluded_ids = (
                    {parent.id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_island_ids = [
                    pid
                    for pid in island_program_ids
                    if pid not in excluded_ids and pid in self.programs
                ]

                if available_island_ids:
                    random_ids = random.sample(
                        available_island_ids, min(remaining, len(available_island_ids))
                    )
                    random_programs = [self.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)

        # Log island isolation info
        logger.debug(
            f"Sampled {len(inspirations)} inspirations from island {parent_island} "
            f"(island has {len(island_programs)} programs total)"
        )

        return inspirations[:n]

    def _remove_program_if_orphaned(self, program_id: str) -> None:
        """
        Remove a program from the population if it is orphaned.

        A program is considered orphaned when it no longer owns a MAP-Elites cell
        in any island's feature map and is not a member of any island. Such a
        program (e.g. one displaced when its cell was improved) can never be
        sampled again but still counts against the population size limit, so it is
        removed from ``self.programs``, the archive and any lingering references.

        Args:
            program_id: ID of the (possibly) orphaned program to check and remove
        """
        if program_id not in self.programs:
            return

        # Still owns a cell in some island? Then it is not orphaned.
        for island_map in self.island_feature_maps:
            if program_id in island_map.values():
                return

        # Still a member of some island? Then it is not orphaned.
        for island in self.islands:
            if program_id in island:
                return

        # Fully orphaned - remove from all remaining structures.
        self._journal_removal(self.programs[program_id], "orphaned")
        del self.programs[program_id]
        self.archive.discard(program_id)
        self._cleanup_stale_island_bests()
        logger.debug(f"Removed orphaned program {program_id} displaced from its cell")

    def _lane_of(self, program: Program) -> Any:
        """The program's lane-group key under the configured lane_metric.

        A rounded-to-int bucket of the metric value; missing/unparseable groups
        as -1. Returns None when no lane_metric is configured (upstream mode).
        """
        return lane_group_of(program, self.lane_metric)

    def lane_report(self) -> Dict[Any, Dict[str, Any]]:
        """Per-group population summary keyed on lane_metric groups.

        For each group: population, cell-owner (elite) count, homeless count,
        best combined_score, and a seeded/evolved split (seeded = injected or
        parentless, evolved = has a parent). Empty when lane_metric is unset --
        there is nothing to group by. Derived on demand; save() embeds a copy so
        a checkpoint's per-group state is readable without loading the database.
        """
        if self.lane_metric is None:
            return {}
        elite_ids = set()
        for island_map in self.island_feature_maps:
            elite_ids.update(island_map.values())

        report: Dict[Any, Dict[str, Any]] = {}
        for program in self.programs.values():
            lane = self._lane_of(program)
            row = report.setdefault(
                lane,
                {
                    "population": 0,
                    "elite": 0,
                    "homeless": 0,
                    "best_combined_score": None,
                    "seeded": 0,
                    "evolved": 0,
                },
            )
            row["population"] += 1
            if program.id in elite_ids:
                row["elite"] += 1
            else:
                row["homeless"] += 1
            score = (program.metrics or {}).get("combined_score")
            if score is not None and (
                row["best_combined_score"] is None or score > row["best_combined_score"]
            ):
                row["best_combined_score"] = score
            if program.metadata.get("injected") or not program.parent_id:
                row["seeded"] += 1
            else:
                row["evolved"] += 1
        return report

    def _group_weights(self, programs: List[Program]) -> List[float]:
        """Per-element parent-draw weights under group-balanced sampling.

        Each group's aggregate weight is ``n ** (1 - gamma)`` for its size ``n``,
        split across members by ascending score rank ``r`` as
        ``(1 - w)/n + w * (r/n) ** p`` (normalised within the group). So a group's
        total draw share is independent of how large its members' scores are, and
        within a group higher scorers are favoured while the floor keeps every
        member reachable. ``w = 0`` reduces to uniform ``n ** -gamma`` per member.
        """
        gamma = float(self.lane_sampling_gamma)
        w = self.lane_rank_weight
        p = self.lane_rank_power
        by_group: Dict[Any, List[int]] = {}
        for i, program in enumerate(programs):
            by_group.setdefault(self._lane_of(program), []).append(i)

        weights = [0.0] * len(programs)
        for idxs in by_group.values():
            n = len(idxs)
            group_total = n ** (1.0 - gamma)
            order = sorted(idxs, key=lambda i: _group_score(programs[i]))  # ascending, stable
            rank = {pos: r for r, pos in enumerate(order, start=1)}
            raw = {pos: (1.0 - w) / n + w * (rank[pos] / n) ** p for pos in idxs}
            norm = sum(raw.values())
            for pos in idxs:
                weights[pos] = group_total * raw[pos] / norm
        return weights

    def _group_weighted_choice(self, valid_ids: List[str]) -> Program:
        """Pick one parent from `valid_ids`, biased toward under-populated groups
        and each group's higher scorers (see `_group_weights`)."""
        programs = [self.programs[pid] for pid in valid_ids]
        weights = self._group_weights(programs)
        return random.choices(programs, weights=weights, k=1)[0]

    def _journal_removal(
        self, program: Program, reason: str, evictor_id: Optional[str] = None
    ) -> None:
        """Record a program removal in the eviction journal (diagnostic only)."""
        self.eviction_journal.append(
            {
                "id": program.id,
                "reason": reason,
                "combined_score": (program.metrics or {}).get("combined_score"),
                "lane": self._lane_of(program),
                "evictor_id": evictor_id,
                "iteration": self.last_iteration,
            }
        )

    def _select_removals(
        self,
        num_to_remove: int,
        non_elite: List[Program],
        elite: List[Program],
        all_programs: List[Program],
    ) -> List[Program]:
        """Choose up to `num_to_remove` programs to evict.

        Upstream (lane_metric is None): worst-first among homeless, then elite.
        Group-balanced (lane_metric set): repeatedly evict the worst homeless
        program from the most-populous lane group, so a group whose scores are
        numerically larger cannot crowd out another group's members. `non_elite`
        must be pre-sorted worst-first. Falls back to elite worst-first only if
        homeless programs run out, exactly as upstream does.
        """
        if self.lane_metric is None:
            chosen = non_elite[:num_to_remove]
            if len(chosen) < num_to_remove:
                chosen = chosen + elite[: num_to_remove - len(chosen)]
            return chosen

        group_pop: Dict[Any, int] = {}
        for program in all_programs:
            lane = self._lane_of(program)
            group_pop[lane] = group_pop.get(lane, 0) + 1
        by_group: Dict[Any, List[Program]] = {}
        for program in non_elite:  # already worst-first, so each list stays worst-first
            by_group.setdefault(self._lane_of(program), []).append(program)

        chosen: List[Program] = []
        for _ in range(num_to_remove):
            candidates = [lane for lane, ps in by_group.items() if ps]
            if not candidates:
                break
            # Most-populous group; ties -> more homeless remaining there.
            lane = max(candidates, key=lambda g: (group_pop[g], len(by_group[g])))
            chosen.append(by_group[lane].pop(0))  # worst-scoring homeless in that group
            group_pop[lane] -= 1
        if len(chosen) < num_to_remove:
            chosen.extend(elite[: num_to_remove - len(chosen)])
        return chosen

    def _enforce_population_limit(
        self,
        exclude_program_id: Optional[str] = None,
        protected_ids: Optional[Set[str]] = None,
    ) -> None:
        """
        Enforce the population size limit by removing worst programs if needed

        Args:
            exclude_program_id: Program ID to never remove (e.g., newly added program)
            protected_ids: Additional program ids to never remove (e.g., other clones
                in an inject() batch, so a later add cannot cull an earlier injection)
        """
        if len(self.programs) <= self.config.population_size:
            return

        # Calculate how many programs to remove
        num_to_remove = len(self.programs) - self.config.population_size

        logger.info(
            f"Population size ({len(self.programs)}) exceeds limit ({self.config.population_size}), removing {num_to_remove} programs"
        )

        # Collect all MAP-Elites cell owners across every island. These "elite"
        # programs represent occupied niches and must be protected from eviction
        # to preserve diversity - a low-scoring cell owner should only be removed
        # after every non-owning (homeless) program has already been removed.
        elite_ids = set()
        for island_map in self.island_feature_maps:
            elite_ids.update(island_map.values())

        # Never remove the best program, the excluded (just-added) program, or any
        # caller-protected id (e.g. earlier clones in an inject() batch).
        protected = {self.best_program_id, exclude_program_id} - {None}
        if protected_ids:
            protected |= set(protected_ids)

        all_programs = list(self.programs.values())

        # Split into non-elite (homeless) and elite (cell owners), each sorted by
        # fitness worst-first. Non-elite programs are removed before elite ones.
        non_elite = sorted(
            [p for p in all_programs if p.id not in elite_ids and p.id not in protected],
            key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
        )
        elite = sorted(
            [p for p in all_programs if p.id in elite_ids and p.id not in protected],
            key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
        )

        programs_to_remove = self._select_removals(num_to_remove, non_elite, elite, all_programs)

        # Remove the selected programs
        for program in programs_to_remove:
            program_id = program.id
            self._journal_removal(program, "population_limit", evictor_id=exclude_program_id)

            # Remove from main programs dict
            if program_id in self.programs:
                del self.programs[program_id]

            # Remove from island feature maps
            for island_idx, island_map in enumerate(self.island_feature_maps):
                keys_to_remove = []
                for key, pid in island_map.items():
                    if pid == program_id:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del island_map[key]

            # Remove from islands
            for island in self.islands:
                island.discard(program_id)

            # Remove from archive
            self.archive.discard(program_id)

            logger.debug(f"Removed program {program_id} due to population limit")

        logger.info(f"Population size after cleanup: {len(self.programs)}")

        # Clean up any stale island best program references after removal
        self._cleanup_stale_island_bests()

    # Island management methods
    def set_current_island(self, island_idx: int) -> None:
        """Set which island is currently being evolved"""
        self.current_island = island_idx % len(self.islands)
        logger.debug(f"Switched to evolving island {self.current_island}")

    def next_island(self) -> int:
        """Move to the next island in round-robin fashion"""
        self.current_island = (self.current_island + 1) % len(self.islands)
        logger.debug(f"Advanced to island {self.current_island}")
        return self.current_island

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:
        """Increment generation counter for an island"""
        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1
        logger.debug(f"Island {idx} generation incremented to {self.island_generations[idx]}")

    def should_migrate(self) -> bool:
        """Check if migration should occur based on generation counters"""
        max_generation = max(self.island_generations)
        return (max_generation - self.last_migration_generation) >= self.migration_interval

    def _select_migrants(self, island_programs: List[Program]) -> List[Program]:
        """Choose an island's migrants: the top `migration_rate` fraction by fitness.

        Upstream (lane_group_migration off): one global fitness ranking, top
        fraction. Group-aware (on): the same fraction taken WITHIN each lane group,
        so a group whose scores are numerically smaller still sends its best across
        islands instead of being crowded out of the slots. Rounding is per group
        (min one per non-empty group), so the migrant total can exceed upstream's.
        """

        def top_fraction(programs: List[Program]) -> List[Program]:
            ordered = sorted(
                programs,
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )
            return ordered[: max(1, int(len(ordered) * self.migration_rate))]

        if not self.lane_group_migration:
            return top_fraction(island_programs)

        by_group: Dict[Any, List[Program]] = {}
        for program in island_programs:
            by_group.setdefault(self._lane_of(program), []).append(program)
        migrants: List[Program] = []
        for group_programs in by_group.values():
            migrants.extend(top_fraction(group_programs))
        return migrants

    def migrate_programs(self) -> None:
        """
        Perform migration between islands

        This should be called periodically to share good solutions between islands
        """
        if len(self.islands) < 2:
            return

        logger.info("Performing migration between islands")

        for i, island in enumerate(self.islands):
            if len(island) == 0:
                continue

            # Select top programs from this island for migration
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]
            if not island_programs:
                continue

            migrants = self._select_migrants(island_programs)

            # Migrate to adjacent islands (ring topology)
            target_islands = [(i + 1) % len(self.islands), (i - 1) % len(self.islands)]

            for migrant in migrants:
                # Prevent re-migration of already migrated programs to avoid exponential duplication.
                # Analysis of actual evolution runs shows this causes severe issues:
                # - Program cb5d07f2 had 183 descendant copies by iteration 850
                # - Program 5645fbd2 had 31 descendant copies
                # - IDs grow exponentially: program_migrant_2_migrant_3_migrant_4_migrant_0...
                #
                # This is particularly problematic for OpenEvolve's MAP-Elites + Island hybrid architecture:
                # 1. All copies have identical code → same complexity/diversity/performance scores
                # 2. They all map to the SAME MAP-Elites cell → only 1 survives, rest discarded
                # 3. Wastes computation evaluating hundreds of identical programs
                # 4. Reduces actual diversity as islands fill with duplicates
                #
                # By preventing already-migrated programs from migrating again, we ensure:
                # - Each program migrates at most once per lineage
                # - True diversity is maintained between islands
                # - Computational resources aren't wasted on duplicates
                # - Aligns with MAP-Elites' one-program-per-cell principle
                if migrant.metadata.get("migrant", False):
                    continue

                for target_island in target_islands:
                    # Skip migration if target island already has a program with identical code
                    # Identical code produces identical metrics, so migration would be wasteful
                    target_island_programs = [
                        self.programs[pid]
                        for pid in self.islands[target_island]
                        if pid in self.programs
                    ]
                    has_duplicate_code = any(p.code == migrant.code for p in target_island_programs)

                    if has_duplicate_code:
                        logger.debug(
                            f"Skipping migration of program {migrant.id[:8]} to island {target_island} "
                            f"(duplicate code already exists)"
                        )
                        continue
                    # Create a copy for migration with simple new UUID
                    import uuid

                    migrant_copy = Program(
                        id=str(uuid.uuid4()),
                        code=migrant.code,
                        changes_description=migrant.changes_description,
                        language=migrant.language,
                        parent_id=migrant.id,
                        generation=migrant.generation,
                        metrics=migrant.metrics.copy(),
                        metadata={**migrant.metadata, "island": target_island, "migrant": True},
                    )

                    # Use add() method to properly handle MAP-Elites deduplication,
                    # feature map updates, and island tracking
                    self.add(migrant_copy, target_island=target_island)

                    # Log migration
                    logger.info(
                        "Program %s migrated to island %d",
                        migrant_copy.id[:8],
                        target_island,
                    )

        # Update last migration generation
        self.last_migration_generation = max(self.island_generations)
        logger.info(f"Migration completed at generation {self.last_migration_generation}")

        # Validate migration results
        self._validate_migration_results()

    def _validate_migration_results(self) -> None:
        """
        Validate migration didn't create inconsistencies

        Checks that:
        1. Program island metadata matches actual island assignment
        2. No programs are assigned to multiple islands
        3. All island best programs exist and are in correct islands
        """
        seen_program_ids = set()

        for i, island in enumerate(self.islands):
            for program_id in island:
                # Check for duplicate assignments
                if program_id in seen_program_ids:
                    logger.error(f"Program {program_id} assigned to multiple islands")
                    continue
                seen_program_ids.add(program_id)

                # Check program exists
                if program_id not in self.programs:
                    logger.warning(f"Island {i} contains nonexistent program {program_id}")
                    continue

                # Check metadata consistency
                program = self.programs[program_id]
                stored_island = program.metadata.get("island")
                if stored_island != i:
                    logger.warning(
                        f"Island mismatch for program {program_id}: "
                        f"in island {i} but metadata says {stored_island}"
                    )

        # Validate island best programs
        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                if best_id not in self.programs:
                    logger.warning(f"Island {i} best program {best_id} does not exist")
                elif best_id not in self.islands[i]:
                    logger.warning(f"Island {i} best program {best_id} not in island")

    def _cleanup_stale_island_bests(self) -> None:
        """
        Remove stale island best program references

        Cleans up references to programs that no longer exist in the database
        or are not actually in their assigned islands.
        """
        cleaned_count = 0

        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                should_clear = False

                # Check if program still exists
                if best_id not in self.programs:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (program deleted)"
                    )
                    should_clear = True
                # Check if program is still in the island
                elif best_id not in self.islands[i]:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (not in island)"
                    )
                    should_clear = True

                if should_clear:
                    self.island_best_programs[i] = None
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale island best program references")

            # Recalculate best programs for islands that were cleared
            for i, best_id in enumerate(self.island_best_programs):
                if best_id is None and len(self.islands[i]) > 0:
                    # Find new best program for this island
                    island_programs = [
                        self.programs[pid] for pid in self.islands[i] if pid in self.programs
                    ]
                    if island_programs:
                        # Sort by fitness and update
                        best_program = max(
                            island_programs,
                            key=lambda p: p.metrics.get(
                                "combined_score", safe_numeric_average(p.metrics)
                            ),
                        )
                        self.island_best_programs[i] = best_program.id
                        logger.debug(f"Recalculated island {i} best program: {best_program.id}")

    def get_island_stats(self) -> List[dict]:
        """Get statistics for each island"""
        stats = []

        for i, island in enumerate(self.islands):
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]

            if island_programs:
                scores = [
                    get_fitness_score(p.metrics, self.config.feature_dimensions)
                    for p in island_programs
                ]

                # The fitness of the program this island ranks best, not the
                # island's highest number: those differ when the top scorer is
                # infeasible, and the id printed beside this is the ranked one.
                ranked = self.exemplar_order(island_programs)
                best_score = (
                    get_fitness_score(ranked[0].metrics, self.config.feature_dimensions)
                    if ranked
                    else 0.0
                )
                avg_score = sum(scores) / len(scores) if scores else 0.0
                diversity = self._calculate_island_diversity(island_programs)
            else:
                best_score = avg_score = diversity = 0.0

            stats.append(
                {
                    "island": i,
                    "population_size": len(island_programs),
                    "best_score": best_score,
                    "average_score": avg_score,
                    "diversity": diversity,
                    "generation": self.island_generations[i],
                    "is_current": i == self.current_island,
                }
            )

        return stats

    def _calculate_island_diversity(self, programs: List[Program]) -> float:
        """Calculate diversity within an island (deterministic version)"""
        if len(programs) < 2:
            return 0.0

        total_diversity = 0
        comparisons = 0

        # Use deterministic sampling instead of random.sample() to ensure consistent results
        sample_size = min(5, len(programs))  # Reduced from 10 to 5

        # Sort programs by ID for deterministic ordering
        sorted_programs = sorted(programs, key=lambda p: p.id)

        # Take first N programs instead of random sampling
        sample_programs = sorted_programs[:sample_size]

        # Limit total comparisons for performance
        max_comparisons = 6  # Maximum comparisons to prevent long delays

        for i, prog1 in enumerate(sample_programs):
            for prog2 in sample_programs[i + 1 :]:
                if comparisons >= max_comparisons:
                    break

                # Use fast approximation instead of expensive edit distance
                diversity = self._fast_code_diversity(prog1.code, prog2.code)
                total_diversity += diversity
                comparisons += 1

            if comparisons >= max_comparisons:
                break

        return total_diversity / max(1, comparisons)

    def _fast_code_diversity(self, code1: str, code2: str) -> float:
        """
        Fast approximation of code diversity using simple metrics

        Returns diversity score (higher = more diverse)
        """
        if code1 == code2:
            return 0.0

        # Length difference (scaled to reasonable range)
        len1, len2 = len(code1), len(code2)
        length_diff = abs(len1 - len2)

        # Line count difference
        lines1 = code1.count("\n")
        lines2 = code2.count("\n")
        line_diff = abs(lines1 - lines2)

        # Simple character set difference
        chars1 = set(code1)
        chars2 = set(code2)
        char_diff = len(chars1.symmetric_difference(chars2))

        # Combine metrics (scaled to match original edit distance range)
        diversity = length_diff * 0.1 + line_diff * 10 + char_diff * 0.5

        return diversity

    def _get_cached_diversity(self, program: Program) -> float:
        """
        Get diversity score for a program using cache and reference set

        Args:
            program: The program to calculate diversity for

        Returns:
            Diversity score (cached or newly computed)
        """
        code_hash = hash(program.code)

        # Check cache first
        if code_hash in self.diversity_cache:
            return self.diversity_cache[code_hash]["value"]

        # Update reference set if needed
        if (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        ):
            self._update_diversity_reference_set()

        # Compute diversity against reference set
        diversity_scores = []
        for ref_code in self.diversity_reference_set:
            if ref_code != program.code:  # Don't compare with itself
                diversity_scores.append(self._fast_code_diversity(program.code, ref_code))

        diversity = (
            sum(diversity_scores) / max(1, len(diversity_scores)) if diversity_scores else 0.0
        )

        # Cache the result with LRU eviction
        self._cache_diversity_value(code_hash, diversity)

        return diversity

    def _update_diversity_reference_set(self) -> None:
        """Update the reference set for diversity calculation"""
        if len(self.programs) == 0:
            return

        # Select diverse programs for reference set
        all_programs = list(self.programs.values())

        if len(all_programs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.code for p in all_programs]
        else:
            # Select programs with maximum diversity
            selected = []
            remaining = all_programs.copy()

            # Start with a random program
            first_idx = random.randint(0, len(remaining) - 1)
            selected.append(remaining.pop(first_idx))

            # Greedily add programs that maximize diversity to selected set
            while len(selected) < self.diversity_reference_size and remaining:
                max_diversity = -1
                best_idx = -1

                for i, candidate in enumerate(remaining):
                    # Calculate minimum diversity to selected programs
                    min_div = float("inf")
                    for selected_prog in selected:
                        div = self._fast_code_diversity(candidate.code, selected_prog.code)
                        min_div = min(min_div, div)

                    if min_div > max_diversity:
                        max_diversity = min_div
                        best_idx = i

                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))

            self.diversity_reference_set = [p.code for p in selected]

        logger.debug(
            f"Updated diversity reference set with {len(self.diversity_reference_set)} programs"
        )

    def _cache_diversity_value(self, code_hash: int, diversity: float) -> None:
        """Cache a diversity value with LRU eviction"""
        # Check if cache is full
        if len(self.diversity_cache) >= self.diversity_cache_size:
            # Remove oldest entry
            oldest_hash = min(self.diversity_cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.diversity_cache[oldest_hash]

        # Add new entry
        self.diversity_cache[code_hash] = {"value": diversity, "timestamp": time.time()}

    def _invalidate_diversity_cache(self) -> None:
        """Invalidate the diversity cache when programs change significantly"""
        self.diversity_cache.clear()
        self.diversity_reference_set = []
        logger.debug("Diversity cache invalidated")

    def _update_feature_stats(self, feature_name: str, value: float) -> None:
        """
        Update statistics for a feature dimension

        Args:
            feature_name: Name of the feature dimension
            value: New value to incorporate into stats
        """
        # A declared-domain axis has a pinned range and is never ratcheted from
        # observed values -- that is the whole point of declaring it.
        if feature_name in self.feature_domains:
            return
        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                "min": value,
                "max": value,
                "values": [],  # Keep recent values for percentile calculation if needed
            }

        stats = self.feature_stats[feature_name]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)

        # Keep recent values for more sophisticated scaling methods
        stats["values"].append(value)
        if len(stats["values"]) > 1000:  # Limit memory usage
            stats["values"] = stats["values"][-1000:]

    def _scale_feature_value(self, feature_name: str, value: float) -> float:
        """
        Scale a feature value according to the configured scaling method

        Args:
            feature_name: Name of the feature dimension
            value: Raw feature value

        Returns:
            Scaled value in range [0, 1]
        """
        # Declared-domain axis: scale against the pinned [min, max], clipped, so
        # the grid never re-folds as observed values drift.
        domain = self.feature_domains.get(feature_name)
        if domain is not None:
            lo, hi = domain[0], domain[1]
            if hi == lo:
                return 0.5
            return min(1.0, max(0.0, (value - lo) / (hi - lo)))

        if feature_name not in self.feature_stats:
            # No stats yet, return normalized by a reasonable default
            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]

        if self.feature_scaling_method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = stats["min"]
            max_val = stats["max"]

            if max_val == min_val:
                return 0.5  # All values are the same

            scaled = (value - min_val) / (max_val - min_val)
            return min(1.0, max(0.0, scaled))  # Ensure in [0, 1]

        elif self.feature_scaling_method == "percentile":
            # Use percentile ranking
            values = stats["values"]
            if not values:
                return 0.5

            # Count how many values are less than or equal to this value
            count = sum(1 for v in values if v <= value)
            percentile = count / len(values)
            return percentile

        else:
            # Default to min-max if unknown method
            return self._scale_feature_value_minmax(feature_name, value)

    def _scale_feature_value_minmax(self, feature_name: str, value: float) -> float:
        """Helper for min-max scaling"""
        if feature_name not in self.feature_stats:
            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]
        min_val = stats["min"]
        max_val = stats["max"]

        if max_val == min_val:
            return 0.5

        scaled = (value - min_val) / (max_val - min_val)
        return min(1.0, max(0.0, scaled))

    def _serialize_feature_stats(self) -> Dict[str, Any]:
        """
        Serialize feature_stats for JSON storage

        Returns:
            Dictionary that can be JSON-serialized
        """
        serialized = {}
        for feature_name, stats in self.feature_stats.items():
            # Convert to JSON-serializable format
            serialized_stats = {}
            for key, value in stats.items():
                if key == "values":
                    # Limit size to prevent excessive memory usage
                    # Keep only the most recent 100 values for percentile calculations
                    if isinstance(value, list) and len(value) > 100:
                        serialized_stats[key] = value[-100:]
                    else:
                        serialized_stats[key] = value
                else:
                    # Convert numpy types to Python native types
                    if hasattr(value, "item"):  # numpy scalar
                        serialized_stats[key] = value.item()
                    else:
                        serialized_stats[key] = value
            serialized[feature_name] = serialized_stats
        return serialized

    def _deserialize_feature_stats(
        self, stats_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Deserialize feature_stats from loaded JSON

        Args:
            stats_dict: Dictionary loaded from JSON

        Returns:
            Properly formatted feature_stats dictionary
        """
        if not stats_dict:
            return {}

        deserialized = {}
        for feature_name, stats in stats_dict.items():
            if isinstance(stats, dict):
                # Ensure proper structure and types
                deserialized_stats = {
                    "min": float(stats.get("min", 0.0)),
                    "max": float(stats.get("max", 1.0)),
                    "values": list(stats.get("values", [])),
                }
                deserialized[feature_name] = deserialized_stats
            else:
                logger.warning(
                    f"Skipping malformed feature_stats entry for '{feature_name}': {stats}"
                )

        return deserialized

    def log_island_status(self) -> None:
        """Log current status of all islands"""
        stats = self.get_island_stats()
        logger.info("Island Status:")
        for stat in stats:
            current_marker = " *" if stat["is_current"] else "  "
            island_idx = stat["island"]
            island_best_id = (
                self.island_best_programs[island_idx]
                if island_idx < len(self.island_best_programs)
                else None
            )
            best_indicator = f" (best: {island_best_id})" if island_best_id else ""
            logger.info(
                f"{current_marker} Island {stat['island']}: {stat['population_size']} programs, "
                f"best={stat['best_score']:.4f}, avg={stat['average_score']:.4f}, "
                f"diversity={stat['diversity']:.2f}, gen={stat['generation']}{best_indicator}"
            )

    # Artifact storage and retrieval methods

    def store_artifacts(self, program_id: str, artifacts: Dict[str, Union[str, bytes]]) -> None:
        """
        Store artifacts for a program

        Args:
            program_id: ID of the program
            artifacts: Dictionary of artifact name to content
        """
        if not artifacts:
            return

        program = self.get(program_id)
        if not program:
            logger.warning(f"Cannot store artifacts: program {program_id} not found")
            return

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"
        if not artifacts_enabled:
            logger.debug("Artifacts disabled, skipping storage")
            return

        # Split artifacts by size
        small_artifacts = {}
        large_artifacts = {}
        size_threshold = getattr(self.config, "artifact_size_threshold", 32 * 1024)  # 32KB default

        for key, value in artifacts.items():
            size = self._get_artifact_size(value)
            if size <= size_threshold:
                small_artifacts[key] = value
            else:
                large_artifacts[key] = value

        # Store small artifacts as JSON
        if small_artifacts:
            program.artifacts_json = json.dumps(small_artifacts, default=self._artifact_serializer)
            logger.debug(f"Stored {len(small_artifacts)} small artifacts for program {program_id}")

        # Store large artifacts to disk
        if large_artifacts:
            artifact_dir = self._create_artifact_dir(program_id)
            program.artifact_dir = artifact_dir
            for key, value in large_artifacts.items():
                self._write_artifact_file(artifact_dir, key, value)
            logger.debug(f"Stored {len(large_artifacts)} large artifacts for program {program_id}")

    def get_artifacts(self, program_id: str) -> Dict[str, Union[str, bytes]]:
        """
        Retrieve all artifacts for a program

        Args:
            program_id: ID of the program

        Returns:
            Dictionary of artifact name to content
        """
        program = self.get(program_id)
        if not program:
            return {}

        artifacts = {}

        # Load small artifacts from JSON
        if program.artifacts_json:
            try:
                small_artifacts = json.loads(program.artifacts_json)
                artifacts.update(small_artifacts)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode artifacts JSON for program {program_id}: {e}")

        # Load large artifacts from disk
        if program.artifact_dir and os.path.exists(program.artifact_dir):
            disk_artifacts = self._load_artifact_dir(program.artifact_dir)
            artifacts.update(disk_artifacts)

        return artifacts

    def _get_artifact_size(self, value: Union[str, bytes]) -> int:
        """Get size of an artifact value in bytes"""
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def _artifact_serializer(self, obj):
        """JSON serializer for artifacts that handles bytes"""
        if isinstance(obj, bytes):
            return {"__bytes__": base64.b64encode(obj).decode("utf-8")}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _artifact_deserializer(self, dct):
        """JSON deserializer for artifacts that handles bytes"""
        if "__bytes__" in dct:
            return base64.b64decode(dct["__bytes__"])
        return dct

    def _create_artifact_dir(self, program_id: str) -> str:
        """Create artifact directory for a program"""
        base_path = getattr(self.config, "artifacts_base_path", None)
        if not base_path:
            base_path = (
                os.path.join(self.config.db_path or ".", "artifacts")
                if self.config.db_path
                else "./artifacts"
            )

        artifact_dir = os.path.join(base_path, program_id)
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir

    def _cleanup_old_artifacts(self, checkpoint_path: str) -> None:
        """
        Remove artifact directories older than the configured retention period.

        Args:
            checkpoint_path: The path of the current checkpoint being saved, which
                             contains the artifacts folder to be cleaned.
        """
        if not self.config.cleanup_old_artifacts:
            return

        artifacts_base_path = os.path.join(checkpoint_path, "artifacts")

        if not os.path.isdir(artifacts_base_path):
            return

        now = time.time()
        retention_seconds = self.config.artifact_retention_days * 24 * 60 * 60
        deleted_count = 0

        logger.debug(f"Starting artifact cleanup in {artifacts_base_path}...")

        for dirname in os.listdir(artifacts_base_path):
            dirpath = os.path.join(artifacts_base_path, dirname)
            if os.path.isdir(dirpath):
                try:
                    dir_mod_time = os.path.getmtime(dirpath)
                    if (now - dir_mod_time) > retention_seconds:
                        shutil.rmtree(dirpath)
                        deleted_count += 1
                        logger.debug(f"Removed old artifact directory: {dirpath}")
                except FileNotFoundError:
                    # Can happen in race conditions; ignore.
                    continue
                except Exception as e:
                    logger.error(f"Error removing artifact directory {dirpath}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old artifact directories.")

    def _write_artifact_file(self, artifact_dir: str, key: str, value: Union[str, bytes]) -> None:
        """Write an artifact to a file"""
        # Sanitize filename
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        if not safe_key:
            safe_key = "artifact"

        file_path = os.path.join(artifact_dir, safe_key)

        try:
            if isinstance(value, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(value)
            elif isinstance(value, bytes):
                with open(file_path, "wb") as f:
                    f.write(value)
            else:
                # Convert to string and write
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(value))
        except Exception as e:
            logger.warning(f"Failed to write artifact {key} to {file_path}: {e}")

    def _load_artifact_dir(self, artifact_dir: str) -> Dict[str, Union[str, bytes]]:
        """Load artifacts from a directory"""
        artifacts = {}

        try:
            for filename in os.listdir(artifact_dir):
                file_path = os.path.join(artifact_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        # Try to read as text first
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except UnicodeDecodeError:
                        # If text fails, read as binary
                        with open(file_path, "rb") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except Exception as e:
                        logger.warning(f"Failed to read artifact file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list artifact directory {artifact_dir}: {e}")

        return artifacts

    def log_prompt(
        self,
        program_id: str,
        template_key: str,
        prompt: Dict[str, str],
        responses: Optional[List[str]] = None,
    ) -> None:
        """
        Log a prompt for a program.
        Only logs if self.config.log_prompts is True.

        Args:
        program_id: ID of the program to log the prompt for
        template_key: Key for the prompt template
        prompt: Prompts in the format {template_key: { 'system': str, 'user': str }}.
        responses: Optional list of responses to the prompt, if available.
        """

        if not self.config.log_prompts:
            return

        if responses is None:
            responses = []
        prompt["responses"] = responses

        if self.prompts_by_program is None:
            self.prompts_by_program = {}

        if program_id not in self.prompts_by_program:
            self.prompts_by_program[program_id] = {}
        self.prompts_by_program[program_id][template_key] = prompt
