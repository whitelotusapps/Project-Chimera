import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from loguru import logger

from .project_paths import PATHS


@dataclass
class ModelConfig:
    use_model: str
    model_name: str
    model_type: str
    model_host: str
    enable_qna: Optional[str] = None
    enable_custom_labels: Optional[str] = None
    server_address: Optional[str] = None
    server_port: Optional[str] = None
    annotators: Optional[str] = None
    pipelineLanguage: Optional[str] = None
    outputFormat: Optional[str] = None
    ner_additional_regexner_mapping_file: Optional[str] = None
    ner_additional_tokensregex_rules_file: Optional[str] = None


@dataclass
class AstrologyConfig:
    planet_and_aspect_orb: str
    natal_date_and_time_of_birth: str
    natal_lat: str
    natal_long: str
    natal_timezone: str
    swiss_eph_path: str
    immanuel_house_system: str
    pos_file: str
    pof_file: str


@dataclass
class AnalysisConfig:
    # Core variables
    idiolect_file_path: Path
    generated_from_corpus_idiotlect_output_path: Path

    analysis_directories_to_process: List[Path]
    analysis_directories_to_ignore: List[Path]
    analysis_output_directory: Path
    analysis_label_file_path: Path
    analysis_questions_file_path: Path
    analysis_source_audio_file_directory: Path

    model_configs: List[ModelConfig]
    astrology_variables: Dict[str, str]
    contractions_dict: Dict[str, str]

    # Paths
    config_path: Path
    log_file: Path

    # Other
    analysis_variables: Dict[str, Any] = field(default_factory=dict)
    desired_date: datetime = datetime(2020, 5, 5)  # default


def load_analysis_config(
    config_path: Optional[Path] = None,
) -> AnalysisConfig:
    # === Set defaults ===
    script_path = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = script_path / "assets" / "config" / "analysis.toml"

    if not config_path.exists():
        logger.opt(colors=True).error(
            f"<RED><white><b>{config_path} file not found.</b></white></RED>"
        )
        sys.exit(1)

    config = toml.load(config_path)
    analysis_variables = config.get("analysis_variables", {})
    astrology_variables = analysis_variables.get("astrology_variables", {})
    # === Environment variables & warnings ===
    warnings.filterwarnings("ignore", category=UserWarning)

    # === Create log file path ===
    log_file = (
        script_path
        / "logs"
        / (datetime.now().strftime("%Y-%m-%d - %H-%M-%S - analysis.log"))
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    analysis_source_audio_file_directory = Path(
        analysis_variables.get("analysis_source_audio_file_directory", "")
    )

    astrology_variables = AstrologyConfig(
        planet_and_aspect_orb=astrology_variables.get("planet_and_aspect_orb", ""),
        natal_date_and_time_of_birth=astrology_variables.get(
            "natal_date_and_time_of_birth", ""
        ),
        natal_lat=astrology_variables.get("natal_lat", ""),
        natal_long=astrology_variables.get("natal_long", ""),
        natal_timezone=astrology_variables.get("natal_timezone", ""),
        swiss_eph_path=astrology_variables.get("swiss_eph_path", ""),
        immanuel_house_system=astrology_variables.get("immanuel_house_system", ""),
        pos_file=PATHS.csv / astrology_variables.get("pos_file", ""),
        pof_file=PATHS.csv / astrology_variables.get("pof_file", ""),
    )

    # === Parse model configs ===
    model_configs = [
        ModelConfig(
            use_model=mc.get("use_model"),
            model_name=mc.get("model_name"),
            model_type=mc.get("model_type"),
            model_host=mc.get("model_host"),
            enable_qna=mc.get("enable_qna"),
            enable_custom_labels=mc.get("enable_custom_labels"),
            server_address=mc.get("server_address"),
            server_port=mc.get("server_port"),
            annotators=mc.get("annotators"),
            pipelineLanguage=mc.get("pipelineLanguage"),
            outputFormat=mc.get("outputFormat"),
            ner_additional_regexner_mapping_file=mc.get(
                "ner_additional_regexner_mapping_file"
            ),
            ner_additional_tokensregex_rules_file=mc.get(
                "ner_additional_tokensregex_rules_file"
            ),
        )
        for mc in analysis_variables.get("model_configs", [])
    ]

    # === Build required paths ===
    required_paths = {
        "idiolect_file_path": Path(analysis_variables.get("idiolect_file_path", "")),
        "generated_from_corpus_idiotlect_output_path": Path(
            analysis_variables.get("generated_from_corpus_idiotlect_output_path", "")
        ),
        "analysis_output_directory": Path(
            analysis_variables.get("analysis_output_directory", "")
        ),
        "analysis_label_file_path": Path(
            analysis_variables.get("analysis_label_file_path", "")
        ),
        "analysis_questions_file_path": Path(
            analysis_variables.get("analysis_questions_file_path", "")
        ),
    }

    # === Validate required paths ===
    file_path_check = True
    for name, path in required_paths.items():
        if not path.exists():
            logger.opt(colors=True).error(
                f"{name}: <RED><white><b>{path}</b></white></RED> path not found."
            )
            file_path_check = False

    # === Validate analysis_directories_to_process ===
    process_dirs = [
        Path(p) for p in analysis_variables.get("analysis_directories_to_process", [])
    ]
    for p in process_dirs:
        if not p.exists():
            logger.opt(colors=True).error(
                f"analysis_directories_to_process: <RED><white><b>{p}</b></white></RED> path not found."
            )
            file_path_check = False

    # === Exit if any paths are invalid ===
    if not file_path_check:
        sys.exit(1)

    # === Build and return config ===
    return AnalysisConfig(
        idiolect_file_path=required_paths["idiolect_file_path"],
        generated_from_corpus_idiotlect_output_path=required_paths[
            "generated_from_corpus_idiotlect_output_path"
        ],
        analysis_directories_to_process=process_dirs,
        analysis_directories_to_ignore=[
            Path(p)
            for p in analysis_variables.get("analysis_directories_to_ignore", [])
        ],
        analysis_output_directory=required_paths["analysis_output_directory"],
        analysis_label_file_path=required_paths["analysis_label_file_path"],
        analysis_questions_file_path=required_paths["analysis_questions_file_path"],
        model_configs=model_configs,
        contractions_dict=analysis_variables.get("contractions_dict", {}),
        log_file=log_file,
        config_path=config_path,
        analysis_variables=analysis_variables,
        astrology_variables=astrology_variables,
        analysis_source_audio_file_directory=analysis_source_audio_file_directory,
    )
