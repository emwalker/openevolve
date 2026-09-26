"""
Configuration handling for OpenEvolve
"""

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import dacite
import yaml

if TYPE_CHECKING:
    from openevolve.llm.base import LLMInterface


_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")  # ${VAR}


def _resolve_env_var(value: Optional[str]) -> Optional[str]:
    """
    Resolve ${VAR} environment variable reference in a string value.
    In current implementation pattern must match the entire string (e.g., "${OPENAI_API_KEY}"),
    not embedded within other text.

    Args:
        value: The string value that may contain ${VAR} syntax

    Returns:
        The resolved value with environment variable expanded, or original value if no match

    Raises:
        ValueError: If the environment variable is referenced but not set
    """
    if value is None:
        return None

    match = _ENV_VAR_PATTERN.match(value)
    if not match:
        return value

    var_name = match.group(1)
    env_value = os.environ.get(var_name)
    if env_value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return env_value


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None

    # LLM provider: "openai" (default), "claude_code" (Claude Code CLI)
    provider: Optional[str] = None

    # Custom LLM client
    init_client: Optional[Callable] = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: Optional[int] = None

    # Reasoning parameters
    reasoning_effort: Optional[str] = None

    # Claude Code CLI budget per call (USD)
    max_budget_usd: Optional[float] = None

    # Manual mode (human-in-the-loop)
    manual_mode: Optional[bool] = None
    _manual_queue_dir: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to resolve ${VAR} env var references in api_key"""
        self.api_key = _resolve_env_var(self.api_key)


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = "https://api.openai.com/v1"

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float | None = 0.7
    top_p: float | None = None
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: List[LLMModelConfig] = field(default_factory=list)

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Backwardes compatibility with primary_model(_weight) options
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    # Reasoning parameters (inherited from LLMModelConfig but can be overridden)
    reasoning_effort: Optional[str] = None

    # Manual mode switch
    manual_mode: bool = False

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        super().__post_init__()  # Resolve ${VAR} in api_key at LLMConfig level

        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # Only validate if this looks like a user config (has some model info)
        # Don't validate during internal/default initialization
        if (
            self.primary_model
            or self.secondary_model
            or self.primary_model_weight
            or self.secondary_model_weight
        ) and not self.models:
            raise ValueError(
                "No LLM models configured. Please specify 'models' array or "
                "'primary_model' in your configuration."
            )

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            # `provider` must be shared: without it a top-level `llm.provider`
            # (e.g. "claude_code") never reaches the per-model configs, every model
            # keeps provider=None, and LLMEnsemble silently routes them all to the
            # OpenAI backend. Propagation uses overwrite=False, so an explicit
            # per-model provider still wins.
            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
            "manual_mode": self.manual_mode,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)

    def rebuild_models(self) -> None:
        """Rebuild the models list after primary_model/secondary_model field changes"""
        # Clear existing models lists
        self.models = []
        self.evaluator_models = []

        # Re-run model generation logic from __post_init__
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            # See the note in __post_init__: `provider` has to be propagated here too,
            # or rebuilding the models drops a top-level provider back to None.
            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
        }
        self.update_model_params(shared_config)


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Large-codebase mode: represent programs in prompts via compact changes descriptions
    programs_as_changes_description: bool = False
    system_message_changes_description: Optional[str] = None
    initial_changes_description: str = ""

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Where the "diverse" exemplar slots are drawn from. Upstream takes the next
    # `num_diverse_programs` by rank after the top ones, which are the top of the
    # same ordering and so are usually near-copies of it. True draws them
    # uniformly from the rest of the group instead, so the slots hold what they
    # are named for. False = upstream.
    diverse_from_island: bool = False

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Optional importable provider of evolution-template variables (module:callable).
    context_provider: Optional[str] = None

    # Optional module:callable returning text appended verbatim to evolution prompts.
    context_appender: Optional[str] = None

    # Recent mutations of the selected parent source supplied to the provider.
    num_recent_attempts: int = 0

    # Meta-prompting
    # Note: meta-prompting features not implemented
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True

    # Feature extraction and program labeling
    suggest_simplification_after_chars: Optional[int] = (
        500  # Suggest simplifying if program exceeds this many characters
    )
    include_changes_under_chars: Optional[int] = (
        100  # Include change descriptions in features if under this length
    )
    concise_implementation_max_lines: Optional[int] = (
        10  # Label as "concise" if program has this many lines or fewer
    )
    comprehensive_implementation_min_lines: Optional[int] = (
        50  # Label as "comprehensive" if program has this many lines or more
    )

    # Diff summary formatting for "Previous Attempts" section
    diff_summary_max_line_len: int = 100  # Truncate lines longer than this
    diff_summary_max_lines: int = 30  # Max lines per SEARCH/REPLACE block

    # Backward compatibility - deprecated
    code_length_threshold: Optional[int] = (
        None  # Deprecated: use suggest_simplification_after_chars
    )


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    # One trial per nominated metadata value, overriding normal breeding eligibility.
    parent_trial_key: Optional[str] = None
    parent_trial_values: List[str] = field(default_factory=list)
    # Note: diversity_metric fixed to "edit_distance"
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    # CRITICAL: For custom dimensions, evaluators must return RAW VALUES, not bin indices
    # Built-in: "complexity", "diversity", "score" (always available)
    # Custom: Any metric from your evaluator (must be continuous values)
    feature_dimensions: List[str] = field(
        default_factory=lambda: ["complexity", "diversity"],
        metadata={
            "help": "List of feature dimensions for MAP-Elites grid. "
            "Built-in dimensions: 'complexity', 'diversity', 'score'. "
            "Custom dimensions: Must match metric names from evaluator. "
            "IMPORTANT: Evaluators must return raw continuous values for custom dimensions, "
            "NOT pre-computed bin indices. OpenEvolve handles all scaling and binning internally."
        },
    )
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Group-balanced population cap. When set to a metric name, the over-cap
    # eviction removes the worst homeless program from the MOST-populous group
    # (programs grouped by int(round(metrics[lane_metric]))) rather than the
    # globally worst homeless program -- so a group whose scores are numerically
    # larger cannot crowd out another group's low-but-in-group-competitive
    # members. The metric is arbitrary and domain-agnostic; None = upstream
    # global worst-first eviction.
    lane_metric: Optional[str] = None

    # Declared feature-axis domains: {axis: [min, max]} pins an axis's scaling
    # range instead of ratcheting it from observed values, so a disappearing
    # category or an outlier cannot stretch the range and re-fold every bin.
    # Axes absent here keep upstream observed-range scaling. Empty = upstream.
    feature_domains: Dict[str, List[float]] = field(default_factory=dict)

    # What to do when a program's metrics lack a declared feature dimension.
    # "error" (default, upstream) raises, which aborts the iteration that
    # submitted the program -- so one malformed metrics dict costs the whole
    # iteration, including cases the evaluator cannot control (a stage timeout
    # returns its own minimal metrics). "floor" instead bins the missing axis at
    # its declared domain minimum (else 0.0) and logs a warning, so the program
    # is still placed and the iteration completes.
    missing_feature_policy: str = "error"

    # Group-balanced parent sampling (requires lane_metric). Each candidate in the
    # sampling pool is weighted n ** -gamma, where n is its group's size in that
    # pool, so a group's aggregate share is n ** (1 - gamma): a group whose scores
    # are numerically larger stops monopolising parent draws simply by being
    # bigger. None (or 0.0) = upstream uniform sampling. gamma = 1 gives every
    # group present an equal share; higher favours under-populated groups.
    lane_sampling_gamma: Optional[float] = None

    # Within-group rank weighting for parent sampling: a member at 1-based
    # ascending score rank r of n gets a share proportional to
    # (1 - w)/n + w * (r/n) ** p of its group's total, so higher-scoring members
    # of a group are drawn more often. w = 0.0 = uniform within group (upstream);
    # the (1 - w)/n floor keeps every member reachable. Only applies when
    # lane_sampling_gamma is set.
    lane_rank_weight: float = 0.0
    lane_rank_power: float = 1.0

    # Scope prompt exemplars (the top/previous lists and the inspiration slots) to
    # the parent's group, with up to lane_cross_inspirations honestly-labelled
    # exemplars drawn from other groups filling the tail inspiration slots. Lets a
    # group whose ceiling sits below the global top-N still see its own exemplars
    # instead of a wall of another group's. False = upstream (global score order).
    # Requires lane_metric.
    lane_prompt_scope: bool = False
    lane_cross_inspirations: int = 1

    # Label each inspiration drawn from outside the parent's group with that
    # group's name (`lane_names`, keyed by group; the key itself when absent) and
    # a note that its score was earned there. Requires lane_metric; False =
    # upstream (inspirations carry no group label).
    lane_label_cross_inspirations: bool = False
    lane_names: Dict[int, str] = field(default_factory=dict)

    # Prefix each program's MAP-Elites cell key with its lane group, giving every
    # group its own grid: two programs identical on all feature axes but in
    # different groups occupy different cells and can never contest one, so a
    # group is isolated structurally rather than via a group-valued feature axis.
    # The grouping metric should then be dropped from feature_dimensions (it is
    # the group prefix now, not grid geometry). Requires lane_metric; False =
    # upstream single shared grid.
    lane_split_grids: bool = False

    # Select island migrants per lane group (each group's top migration_rate
    # fraction migrates) instead of by global score, so a group whose scores are
    # numerically smaller still propagates its best across islands rather than
    # being shut out of the migration slots by a denser group. Rounding is
    # per-group (minimum one migrant per non-empty group), so totals can differ
    # slightly from upstream. Requires lane_metric; False = upstream global-score
    # selection.
    lane_group_migration: bool = False

    # Constraint handling. When `feasibility_metric` names a metric, a program
    # whose value is <= 0 is archived with its lineage but is never drawn as a
    # parent, never reported as best, and never preferred over a feasible one
    # however it scores; the constraint gates, it does not rank. A program that
    # carries no measurement of it is feasible -- absent is not in breach.
    # `violation_metric` orders the infeasible ones (lower is closer): until the
    # population holds anything feasible, selection narrows to the closest half,
    # which keeps a gradient without collapsing onto one lineage. Once something
    # clears, `feasibility_min_pool` keeps that many of the closest infeasible
    # candidates breedable, so a narrow miss is not walled out by the first
    # arrival. Infeasible programs still appear in prompts -- a near-miss's ideas
    # are worth showing -- but after every feasible one and labelled as failing.
    # The metrics are arbitrary and domain-agnostic; None = upstream, where every
    # program is feasible and none of this applies.
    feasibility_metric: Optional[str] = None
    violation_metric: Optional[str] = None
    feasibility_min_pool: int = 0

    # Artifact keys copied onto a child's metadata when it is created. Metrics are
    # coerced to float, so an evaluator that identifies a program by a STRING --
    # a hash of what it does rather than of its text -- has no way to hand that
    # identity to the engine; an artifact does. Empty = upstream, where metadata
    # carries only the engine's own provenance.
    metadata_from_artifacts: List[str] = field(default_factory=list)

    # Metadata key identifying what a program IS, for the exemplar lists. When
    # set, each list of programs shown to the generator keeps one member per
    # distinct value, so a population holding many copies of one thing does not
    # spend every prompt slot on it. A program carrying no value is never
    # collapsed. The key is arbitrary and domain-agnostic; None = upstream.
    dedup_key: Optional[str] = None

    # A metric marking a program the evaluator does not want stored. When set, a
    # program whose value is truthy is not added -- no entry in the population,
    # no island, no archive, no cell -- and `add` returns its id as it does for a
    # novelty rejection. For an evaluator that can tell it has already measured
    # this exact program; None = upstream, where everything is stored.
    reject_metric: Optional[str] = None

    # Break equal-fitness ties on a metric, lowest first, instead of on arrival
    # order. Where fitness moves in coarse steps, exact ties are common and which
    # program a cell keeps is otherwise whichever arrived first. A program with no
    # usable value never displaces one that has it. None = upstream (arrival).
    tiebreak_metric: Optional[str] = None

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42
    # Reseed the sampling generator on load() from `random_seed` folded with the
    # loaded database's last iteration, so consecutive resumed runs draw fresh
    # numbers; a fresh start is unchanged. False = upstream (every run, resumed or
    # not, opens on `random_seed`'s stream).
    resume_seed_folds_iteration: bool = False

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30
    max_snapshot_artifacts: Optional[int] = (
        100  # Max artifacts in worker snapshots (None=unlimited)
    )

    novelty_llm: Optional["LLMInterface"] = None
    embedding_model: Optional[str] = None
    similarity_threshold: float = 0.99


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    # Note: resource limits not implemented
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    # Note: distributed evaluation not implemented
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program


@dataclass
class EvolutionTraceConfig:
    """Configuration for evolution trace logging"""

    enabled: bool = False
    format: str = "jsonl"  # Options: "jsonl", "json", "hdf5"
    include_code: bool = False
    include_prompts: bool = True
    output_path: Optional[str] = None
    buffer_size: int = 10
    compress: bool = False


@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None
    file_suffix: str = ".py"

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    evolution_trace: EvolutionTraceConfig = field(default_factory=EvolutionTraceConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"

    # Early stopping settings
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"

    # Parallel controller settings
    max_tasks_per_child: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        config_path = Path(path).resolve()
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = cls.from_dict(config_dict)

        # Resolve template_dir relative to config file location
        if config.prompt.template_dir:
            template_path = Path(config.prompt.template_dir)
            if not template_path.is_absolute():
                config.prompt.template_dir = str((config_path.parent / template_path).resolve())

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        if "diff_pattern" in config_dict:
            try:
                re.compile(config_dict["diff_pattern"])
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in diff_pattern: {e}")

        # Remove None values for temperature and top_p to avoid dacite type errors;
        # alternatively, pass check_types=False to dacite.from_dict, but that can hide other issues
        if "llm" in config_dict:
            if "temperature" in config_dict["llm"] and config_dict["llm"]["temperature"] is None:
                del config_dict["llm"]["temperature"]
            if "top_p" in config_dict["llm"] and config_dict["llm"]["top_p"] is None:
                del config_dict["llm"]["top_p"]

        config: Config = dacite.from_dict(
            data_class=cls,
            data=config_dict,
            config=dacite.Config(
                cast=[List, Union],
                forward_references={"LLMInterface": Any},
            ),
        )

        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed

        if config.prompt.programs_as_changes_description and not config.diff_based_evolution:
            raise ValueError(
                "prompt.programs_as_changes_description=true requires diff_based_evolution=true "
                "(full rewrites cannot reliably update code and changes_description together)"
            )

        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
