"""Unit tests for scene file management."""

from polyumi_pi.files.scene import SceneFiles
from polyumi_pi.files.session import SessionFiles


def test_scene_start_propagates_into_session_metadata(tmp_path):
    """A scene's start time reaches the host on metadata.json, the only file fetch transfers."""
    scene = SceneFiles.create(base_dir=tmp_path)
    session = scene.create_session()
    session.metadata.to_file()

    assert scene.started_at is not None
    assert session.metadata.scene_started_at == scene.started_at
    # ...and survives the JSON round trip the fetch/catalog path reads it back through.
    assert SessionFiles.from_file(session.path).metadata.scene_started_at == scene.started_at
    # The scene starts before its first session, which is the whole point: the gap is setup time.
    assert scene.started_at <= session.metadata.created_at


def test_scene_reload_keeps_its_start_time(tmp_path):
    """SceneFiles.from_file recovers started_at, so a reloaded scene still stamps its sessions."""
    scene = SceneFiles.create(base_dir=tmp_path)
    scene.create_session().metadata.to_file()

    reloaded = SceneFiles.from_file(scene.path)
    assert reloaded.started_at == scene.started_at
    assert reloaded.create_session().metadata.scene_started_at == scene.started_at
