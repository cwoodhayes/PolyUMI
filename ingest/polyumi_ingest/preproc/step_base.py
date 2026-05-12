"""Shared preprocessing step base class and pipeline helpers."""

from __future__ import annotations

import logging
import pathlib
import shutil
from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import zarr

_PS = TypeVar('_PS', bound='PreprocessingStep')

from polyumi_ingest.pzarr.scene_files import SceneFiles

log = logging.getLogger(__name__)

PREPROCESSING_STEPS: dict[int, type[PreprocessingStep]] = {}


def register_preprocessing_step(step_number: int, step_name: str):
    """Register a preprocessing step class with explicit metadata."""

    def decorator(cls: type[_PS]) -> type[_PS]:
        if step_number in PREPROCESSING_STEPS:
            raise ValueError(f'Duplicate preprocessing step: {step_number}')
        cls.step_number = step_number  # type: ignore[attr-defined]
        cls.step_name = step_name  # type: ignore[attr-defined]
        PREPROCESSING_STEPS[step_number] = cls
        return cls

    return decorator


def available_preprocessing_steps() -> list[type[PreprocessingStep]]:
    """Return registered preprocessing steps in execution order."""
    return [PREPROCESSING_STEPS[k] for k in sorted(PREPROCESSING_STEPS)]


def _scene_dirs(recordings_dir: pathlib.Path) -> list[pathlib.Path]:
    recordings_dir = recordings_dir.resolve()
    if not recordings_dir.is_dir():
        raise FileNotFoundError(f'Recordings directory not found: {recordings_dir}')
    return sorted(p for p in recordings_dir.iterdir() if p.is_dir() and p.name.startswith('scene_'))


def _preprocessing_steps_done(root: zarr.Group) -> list[int]:
    raw = root.attrs.get('preprocessing_steps', [])
    if not isinstance(raw, list):
        return []
    try:
        return [int(step) for step in raw if isinstance(step, (int, float, str))]
    except (ValueError, TypeError):
        log.warning(f'Invalid preprocessing_steps attribute: {raw}')
        return []


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

    step_number: int
    step_name: str

    @abstractmethod
    def run_step(self, scene_zarr: pathlib.Path) -> None:
        """Mutate a scene.zarr store in place."""

    def run(self, scene_path: pathlib.Path, copy: bool = False, force: bool = False) -> pathlib.Path:
        """Run the step on a scene directory or scene.zarr path."""
        scene_zarr = SceneFiles.resolve_zarr_path(scene_path)
        if not scene_zarr.exists():
            raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

        target_zarr = scene_zarr
        if copy:
            target_zarr = scene_zarr.parent / f'scene_pp{self.step_number}.zarr'
            if target_zarr.exists():
                if not force:
                    raise FileExistsError(f'Preprocessed scene already exists: {target_zarr}')
                shutil.rmtree(target_zarr)
            shutil.copytree(scene_zarr, target_zarr)

        self.run_step(target_zarr)
        root = zarr.open_group(str(target_zarr), mode='a')
        _mark_preprocessing_step(root, self.step_number)
        return target_zarr


def run_preprocessing(
    scene_path: pathlib.Path,
    step_number: int | None = None,
    copy: bool = False,
    force: bool = False,
) -> pathlib.Path:
    """Run one preprocessing step or the full pipeline on a scene."""
    scene_zarr = SceneFiles.resolve_zarr_path(scene_path)
    if not scene_zarr.exists():
        raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

    root = zarr.open_group(str(scene_zarr), mode='a')
    completed_steps = set(_preprocessing_steps_done(root))
    step_numbers = [step_number] if step_number is not None else sorted(PREPROCESSING_STEPS)

    current_path = scene_path
    for number in step_numbers:
        try:
            step_cls = PREPROCESSING_STEPS[number]
        except KeyError:
            raise KeyError(f'Unknown preprocessing step: {number}')
        if number in completed_steps and not force:
            log.info(f'Skipping {scene_zarr.name}: step {number} already complete')
            continue
        log.info(f'Running step {number} ({step_cls.step_name}) on {scene_zarr.name}')
        step = step_cls()
        current_path = step.run(current_path, copy=copy, force=force)
        scene_zarr = SceneFiles.resolve_zarr_path(current_path)
        root = zarr.open_group(str(scene_zarr), mode='a')
        _mark_preprocessing_step(root, number)
        completed_steps = set(_preprocessing_steps_done(root))
        if step_number is None:
            copy = False

    return SceneFiles.resolve_zarr_path(current_path)


def run_preprocessing_on_recordings(
    recordings_dir: pathlib.Path,
    step_number: int | None = None,
    copy: bool = False,
    force: bool = False,
) -> list[pathlib.Path]:
    """Run preprocessing on every scene under recordings_dir."""
    outputs: list[pathlib.Path] = []
    for scene_dir in _scene_dirs(recordings_dir):
        zarr_path = SceneFiles.resolve_zarr_path(scene_dir)
        if not zarr_path.exists():
            log.info(f'Skipping {scene_dir.name}: no scene.zarr found')
            continue
        outputs.append(run_preprocessing(scene_dir, step_number=step_number, copy=copy, force=force))
    return outputs
