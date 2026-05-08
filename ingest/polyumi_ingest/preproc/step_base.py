"""Shared preprocessing step base class and pipeline helpers."""

from __future__ import annotations

import logging
import pathlib
import shutil
from abc import ABC, abstractmethod

import numpy as np
import zarr

from polyumi_ingest.pzarr.scene_files import SceneFiles

log = logging.getLogger(__name__)

PREPROCESSING_STEPS: dict[int, type[PreprocessingStep]] = {}
_PREPROCESSING_STEP_INFO: dict[type[PreprocessingStep], tuple[int, str]] = {}


def register_preprocessing_step(step_number: int, step_name: str):
    """Register a preprocessing step class with explicit metadata."""

    def decorator(cls: type[PreprocessingStep]) -> type[PreprocessingStep]:
        if step_number in PREPROCESSING_STEPS:
            raise ValueError(f'Duplicate preprocessing step: {step_number}')
        PREPROCESSING_STEPS[step_number] = cls
        _PREPROCESSING_STEP_INFO[cls] = (step_number, step_name)
        return cls

    return decorator


def _step_metadata(step_cls: type[PreprocessingStep]) -> tuple[int, str]:
    try:
        return _PREPROCESSING_STEP_INFO[step_cls]
    except KeyError as exc:
        raise KeyError(f'Unregistered preprocessing step class: {step_cls.__name__}') from exc


def available_preprocessing_steps() -> list[type[PreprocessingStep]]:
    """Return registered preprocessing steps in execution order."""
    return [PREPROCESSING_STEPS[k] for k in sorted(PREPROCESSING_STEPS)]


def get_preprocessing_step(step_number: int) -> type[PreprocessingStep]:
    """Return the registered preprocessing step for step_number."""
    try:
        return PREPROCESSING_STEPS[step_number]
    except KeyError as exc:
        raise KeyError(f'Unknown preprocessing step: {step_number}') from exc


def _scene_dirs(recordings_dir: pathlib.Path) -> list[pathlib.Path]:
    recordings_dir = recordings_dir.resolve()
    if not recordings_dir.is_dir():
        raise FileNotFoundError(f'Recordings directory not found: {recordings_dir}')
    return sorted(p for p in recordings_dir.iterdir() if p.is_dir() and p.name.startswith('scene_'))


def _preprocessing_steps_done(root: zarr.Group) -> list[int]:
    raw = root.attrs.get('preprocessing_steps', [])
    if raw is None:
        return []
    return [int(step) for step in raw]


def _mark_preprocessing_step(root: zarr.Group, step_number: int) -> None:
    steps = _preprocessing_steps_done(root)
    if step_number not in steps:
        steps.append(step_number)
        steps.sort()
    root.attrs['preprocessing_steps'] = steps


def _write_scalar(group: zarr.Group, name: str, value: float | int) -> None:
    """Write a scalar zarr array, replacing any existing value."""
    if name in group:
        del group[name]
    group.create_array(name, data=np.array(value))


class PreprocessingStep(ABC):
    """Base class for a single preprocessing step."""

    @property
    def step_number(self) -> int:
        """Return this step's numeric identifier."""
        return _step_metadata(type(self))[0]

    @property
    def step_name(self) -> str:
        """Return this step's display name."""
        return _step_metadata(type(self))[1]

    @abstractmethod
    def run_step(self, scene_zarr: pathlib.Path) -> None:
        """Mutate a scene.zarr store in place."""

    def run(self, scene_path: pathlib.Path, copy: bool = False) -> pathlib.Path:
        """Run the step on a scene directory or scene.zarr path."""
        scene_zarr = SceneFiles.resolve_zarr_path(scene_path)
        if not scene_zarr.exists():
            raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

        target_zarr = scene_zarr
        if copy:
            target_zarr = scene_zarr.parent / f'scene_pp{self.step_number}.zarr'
            if target_zarr.exists():
                raise FileExistsError(f'Preprocessed scene already exists: {target_zarr}')
            shutil.copytree(scene_zarr, target_zarr)

        self.run_step(target_zarr)
        return target_zarr


def run_preprocessing(scene_path: pathlib.Path, step_number: int | None = None, copy: bool = False) -> pathlib.Path:
    """Run one preprocessing step or the full pipeline on a scene."""
    scene_zarr = SceneFiles.resolve_zarr_path(scene_path)
    if not scene_zarr.exists():
        raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

    root = zarr.open_group(str(scene_zarr), mode='a')
    completed_steps = set(_preprocessing_steps_done(root))

    if step_number is not None:
        step_numbers = [step_number]
    else:
        step_numbers = sorted(PREPROCESSING_STEPS.keys())

    current_path = scene_path
    for number in step_numbers:
        step_cls = get_preprocessing_step(number)
        if number in completed_steps:
            log.info(f'Skipping {scene_zarr.name}: step {number} already complete')
            continue
        _, step_name = _step_metadata(step_cls)
        log.info(f'Running step {number} ({step_name}) on {scene_zarr.name}')
        step = step_cls()
        current_path = step.run(current_path, copy=copy)
        scene_zarr = SceneFiles.resolve_zarr_path(current_path)
        root = zarr.open_group(str(scene_zarr), mode='a')
        completed_steps = set(_preprocessing_steps_done(root))
        if step_number is None:
            copy = False

    return SceneFiles.resolve_zarr_path(current_path)


def run_preprocessing_on_recordings(recordings_dir: pathlib.Path, step_number: int | None = None, copy: bool = False) -> list[pathlib.Path]:
    """Run preprocessing on every scene under recordings_dir."""
    outputs: list[pathlib.Path] = []
    for scene_dir in _scene_dirs(recordings_dir):
        zarr_path = SceneFiles.resolve_zarr_path(scene_dir)
        if not zarr_path.exists():
            log.info(f'Skipping {scene_dir.name}: no scene.zarr found')
            continue
        outputs.append(run_preprocessing(scene_dir, step_number=step_number, copy=copy))
    return outputs
