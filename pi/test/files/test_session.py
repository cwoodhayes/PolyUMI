"""Unit tests for session file management."""

from polyumi_pi.files.session import SessionFiles


def test_session_create_and_read_roundtrip(tmp_path):
    """Creating a session writes metadata and can be loaded back."""
    created = SessionFiles.create(base_dir=tmp_path)
    expected_folder_name = created.metadata.created_at.astimezone().strftime(
        'session_%Y-%m-%d_%H-%M-%S'
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
