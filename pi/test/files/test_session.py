"""Unit tests for session file management."""

from polyumi_pi.files.scene import SceneFiles
from polyumi_pi.files.session import SessionFiles


def test_session_create_and_read_roundtrip(tmp_path):
    """Creating a session writes metadata and can be loaded back."""
    created = SessionFiles.create(base_dir=tmp_path)
    expected_folder_name = created.metadata.created_at.astimezone().strftime(
        'session_%Y-%m-%d_%H-%M-%S_' + created.metadata.session_id[:4]
    )

    assert created.path.is_dir()
    assert created.path.name == expected_folder_name
    assert created.metadata.path == created.path / 'metadata.json'
    assert created.metadata.path.is_file()

    loaded = SessionFiles.from_file(created.path)

    assert loaded.path == created.path
    assert loaded.metadata.path == created.metadata.path
    assert loaded.metadata.session_id == created.metadata.session_id
    assert loaded.metadata.created_at == created.metadata.created_at


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
