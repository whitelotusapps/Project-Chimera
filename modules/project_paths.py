from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    # Root
    root: Path = Path(__file__).resolve().parent.parent

    # Core folders
    assets: Path = root / "assets"
    logs: Path = root / "logs"
    modules: Path = root / "modules"
    output: Path = root / "output"

    # Asset subfolders
    category_lists: Path = assets / "category_lists"
    config: Path = assets / "config"
    corpus: Path = assets / "corpus"
    csv: Path = assets / "CSV"
    fonts: Path = assets / "fonts"
    idiolect: Path = assets / "idiolect"
    questions: Path = assets / "questions"
    temp: Path = assets / "temp"

    # Logs subfolders
    logs_corpus: Path = logs / "corpus"

    # Output subfolders
    output_corpus: Path = output / "corpus"


# Single instance to import
PATHS = ProjectPaths()
