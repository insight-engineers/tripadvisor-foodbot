import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STREAMLIT_FILE_PATH = os.getenv("STREAMLIT_FILE_PATH", default="app.py")


def get_file_paths() -> list[str]:
    page_folder = Path(STREAMLIT_FILE_PATH).parent / "pages"
    if not page_folder.exists():
        return [STREAMLIT_FILE_PATH]
    page_files = page_folder.glob("*.py")
    file_paths = [str(file.absolute().resolve()) for file in page_files]
    return [STREAMLIT_FILE_PATH] + file_paths


def pytest_generate_tests(metafunc):
    if "file_path" in metafunc.fixturenames:
        metafunc.parametrize("file_path", get_file_paths(), ids=lambda x: x.split("/")[-1])


def test_app_runs(file_path):
    app_test = AppTest.from_file(file_path, default_timeout=30)
    app_test.session_state.skip_agent_init = True
    app_test.run()
    assert not app_test.exception, f"Exception raised in {file_path}: {app_test.exception}"
