import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import toml
from loguru import logger


@dataclass
class WordCloudConfig:
    top_n_words: int
    custom_stopwords: Set[str]


@dataclass
class SearchAndReplacePair:
    search: str
    replace: str


@dataclass
class TranscriptionConfig:
    # Core variables
    transcription_model: str
    file_extensions_to_transcribe: List[str]
    file_extensions_pattern: str
    corpus_extensions_pattern: str
    save_to_gpt4all_localdocs: str
    faster_whisper_binary_path: Path

    # Paths
    audio_file_directories_to_process: List[Path]
    audio_file_directories_to_ignore: List[Path]
    transcription_output_path: Path
    search_and_replace_transcription_output_path: Path
    gpt4all_localdocs_path: Path
    config_path: Path
    faster_whisper_model_path: Path

    # Other
    search_and_replace_pairs: List[SearchAndReplacePair]
    word_cloud: WordCloudConfig
    log_file: Path
    currently_transcribed_files: List
    corpus_extensions: List
    desired_date: datetime = datetime(2020, 5, 5)  # default

    # Default arguments must come last
    transcription_variables: Dict[str, Any] = field(default_factory=dict)


def load_transcription_config(
    config_path: Optional[Path] = None,
) -> TranscriptionConfig:
    # === Set defaults ===
    script_path = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = script_path / "assets" / "config" / "transcription.toml"

    if not config_path.exists():
        logger.opt(colors=True).error(
            f"<RED><white><b>{config_path} file not found.</b></white></RED>"
        )
        sys.exit(1)

    config = toml.load(config_path)
    transcription_variables = config.get("transcription_variables", {})

    # === Environment variables & warnings ===
    os.environ["CT2_VERBOSE"] = "1"
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="sklearn.feature_extraction.text"
    )

    # === Create log file path ===
    log_file = (
        script_path
        / "logs"
        / (datetime.now().strftime("%Y-%m-%d - %H-%M-%S - transcriptions.log"))
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # === Build faster_whisper_binary_path ===
    base_path = Path(transcription_variables.get("faster_whisper_base_path", ""))
    version = transcription_variables.get("faster_whisper_version", "large-v2")
    binary_name = transcription_variables.get("faster_whisper_binary_name", "")
    faster_whisper_binary_path = (base_path / version / binary_name).resolve()
    faster_whisper_model_path = transcription_variables.get(
        "faster_whisper_model_path", ""
    )

    # === Parse word cloud ===
    word_cloud_data = transcription_variables.get("word_cloud_variables", [{}])[0]
    word_cloud = WordCloudConfig(
        top_n_words=word_cloud_data.get("top_n_words", 300),
        custom_stopwords=set(word_cloud_data.get("custom_stopwords", [])),
    )

    # === Parse search & replace ===
    search_and_replace_pairs = [
        SearchAndReplacePair(search=pair.get("search"), replace=pair.get("replace"))
        for pair in transcription_variables.get("search_and_replace_pairs", [])
    ]

    # === File extensions regex ===
    file_extensions = transcription_variables.get("file_extensions_to_transcribe", [])
    file_extensions_pattern = "|".join(ext.lstrip(".") for ext in file_extensions)

    corpus_extensions = transcription_variables.get(
        "corpus_extensions", [".mp3", ".flac"]
    )
    corpus_extensions_pattern = "|".join(ext.lstrip(".") for ext in corpus_extensions)

    # === Required paths ===
    required_paths = {
        "faster_whisper_binary_path": faster_whisper_binary_path,
        "transcription_output_path": Path(
            transcription_variables.get("transcription_output_path", "")
        ),
        "search_and_replace_transcription_output_path": Path(
            transcription_variables.get(
                "search_and_replace_transcription_output_path", ""
            )
        ),
        "gpt4all_localdocs_path": Path(
            transcription_variables.get("gpt4all_localdocs_path", "")
        ),
    }

    # === Get currently transcribed files ===
    json_dir = required_paths["search_and_replace_transcription_output_path"] / "JSON"

    if json_dir.exists() and json_dir.is_dir():
        currently_transcribed_files = [
            f.name for f in json_dir.iterdir() if f.is_file()
        ]
    else:
        logger.opt(colors=True).error(
            f"search_and_replace_transcription_output_path: <RED><white><b>{json_dir}</b></white></RED> path not found."
        )
        file_path_check = False
        currently_transcribed_files = []

    # Validate required paths
    file_path_check = True
    for name, path in required_paths.items():
        if not path.exists():
            logger.opt(colors=True).error(
                f"{name}: <RED><white><b>{path}</b></white></RED> path not found."
            )
            file_path_check = False

    # === Validate audio_file_directories_to_process ===
    process_dirs = [
        Path(p)
        for p in transcription_variables.get("audio_file_directories_to_process", [])
    ]
    for p in process_dirs:
        if not p.exists():
            logger.opt(colors=True).error(
                f"audio_file_directories_to_process: <RED><white><b>{p}</b></white></RED> path not found."
            )
            file_path_check = False

    # === Exit if any paths are invalid ===
    if not file_path_check:
        sys.exit(1)

    # === Build and return config ===
    return TranscriptionConfig(
        transcription_model=transcription_variables.get(
            "transcription_model", "large-v2"
        ),
        file_extensions_to_transcribe=file_extensions,
        file_extensions_pattern=file_extensions_pattern,
        save_to_gpt4all_localdocs=transcription_variables.get(
            "save_to_gpt4all_localdocs", "no"
        ),
        faster_whisper_binary_path=faster_whisper_binary_path,
        audio_file_directories_to_process=process_dirs,
        audio_file_directories_to_ignore=[
            Path(p)
            for p in transcription_variables.get("audio_file_directories_to_ignore", [])
        ],
        transcription_output_path=required_paths["transcription_output_path"],
        search_and_replace_transcription_output_path=required_paths[
            "search_and_replace_transcription_output_path"
        ],
        gpt4all_localdocs_path=required_paths["gpt4all_localdocs_path"],
        search_and_replace_pairs=search_and_replace_pairs,
        word_cloud=word_cloud,
        log_file=log_file,
        transcription_variables=transcription_variables,
        config_path=config_path,
        currently_transcribed_files=currently_transcribed_files,
        corpus_extensions=corpus_extensions,
        corpus_extensions_pattern=corpus_extensions_pattern,
        faster_whisper_model_path=faster_whisper_model_path,
    )
