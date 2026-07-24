"""
Phase 2 steering toolkit — public API.

Submodules are internal (constrastive.py, extract.py, schedules.py,
positions.py, directions.py, calibrate.py, select.py, intervene.py,
analysis.py); scripts should import from `properties.steering` directly.

`schedules.from_spec` and `positions.from_spec` share a name, so they are
re-exported here as `schedule_from_spec` / `position_from_spec`.
"""
from __future__ import annotations

from .constrastive import (
    ContrastiveSet,
    atom_count_bins,
    match_atom_counts,
    gate_a_check,
    decile_bins,
)
from .extract import (
    SiteReducer,
    PrefixReducer,
    PooledReducer,
    MaskReducer,
    collect_activations,
    diff_of_means,
    ridge_probe,
    bootstrap_directions,
    probe_site_designs,
    interpret_probe,
    projection_check,
)
from .schedules import (
    Schedule,
    AllSteps,
    FirstK,
    LastK,
    EveryK,
    StepWindow,
    NoSteps,
    from_spec as schedule_from_spec,
    sweep_specs,
)
from .positions import (
    PositionSet,
    AllPositions,
    PrefixOnly,
    BodyOnly,
    MaskedOnly,
    FrozenOnly,
    from_spec as position_from_spec,
    SWEEP_SPECS,
)
from .directions import (
    DirectionArtifact,
    DirectionStore,
)
from .calibrate import (
    save_residual_stats,
    load_residual_stats,
    fit_residual_stats,
    normalise_alpha,
    PiecewiseAlpha,
    fit_piecewise_alpha,
    monotonicity,
)
from .select import (
    reference_fingerprints,
    sweep_candidates,
    select_operating_point,
    heatmap_data,
    plot_heatmap,
)
from .intervene import (
    AdditiveSteer,
    ProjectiveSteer,
    ComposedSteer,
    NullSteer,
    build_steer,
)
from .analysis import (
    effective_dim,
    direction_similarity,
    per_decile_directions,
    offtarget_matrix,
    fragment_frequency_shift,
)

__all__ = [
    "ContrastiveSet", "atom_count_bins", "match_atom_counts", "gate_a_check",
    "decile_bins",
    "SiteReducer", "PrefixReducer", "PooledReducer", "MaskReducer",
    "collect_activations", "diff_of_means", "ridge_probe",
    "bootstrap_directions", "probe_site_designs", "interpret_probe",
    "projection_check",
    "Schedule", "AllSteps", "FirstK", "LastK", "EveryK", "StepWindow",
    "NoSteps", "schedule_from_spec", "sweep_specs",
    "PositionSet", "AllPositions", "PrefixOnly", "BodyOnly", "MaskedOnly",
    "FrozenOnly", "position_from_spec", "SWEEP_SPECS",
    "DirectionArtifact", "DirectionStore",
    "save_residual_stats", "load_residual_stats", "fit_residual_stats",
    "normalise_alpha", "PiecewiseAlpha", "fit_piecewise_alpha",
    "monotonicity",
    "reference_fingerprints", "sweep_candidates", "select_operating_point",
    "heatmap_data", "plot_heatmap",
    "AdditiveSteer", "ProjectiveSteer", "ComposedSteer", "NullSteer",
    "build_steer",
    "effective_dim", "direction_similarity", "per_decile_directions",
    "offtarget_matrix", "fragment_frequency_shift",
]
