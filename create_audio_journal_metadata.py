# THIS FILE EMPLOYS MULTIPLE AI MODELS TO ENRICH THE TRANSCRIPTION JSON WITH ADDITIONAL FEATURES

#####################################################################################################################################
# NATIVE MODULES
import csv
import json
import os
import re
from dataclasses import asdict, is_dataclass

# import datetime
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger
from rich.console import Console

#####################################################################################################################################
# HELPER MODULES
from tqdm.auto import tqdm

#####################################################################################################################################
# AI SPECIFIC MODULES
from transformers import logging as hf_logging

from modules.ai_model_loading import load_models

#####################################################################################################################################
# PROJECT SPECIFIC MODULES
from modules.ai_models_output import generate_ai_model_results
from modules.analysis_config_loader import load_analysis_config
from modules.date_functions import extract_date_time_from_json_filename
from modules.generate_chunk_profections import calculate_current_profections
from modules.generate_chunk_root import generate_chunk_root
from modules.generate_chunk_transits import calculate_chunk_transits
from modules.generate_chunk_zrs import generate_zrs_data
from modules.generate_file_chunks import calculate_duration, create_chunk_data
from modules.generate_tags import generate_chunk_tags
from modules.helper_functions import insert_keys, print_table
from modules.idioms_and_beliefs import load_iodlect
from modules.audio_file_metadata import generate_audio_metadata

analysis_config = load_analysis_config()

#####################################################################################################################################
# Set the logging level to error to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
hf_logging.set_verbosity_error()

os.system("clear")

# Create a rich console
console = Console()

# Create a console for printing the config table to our logfile
file_console = Console(color_system=None)


#####################################################################################################################################


#####################################################################################################################################
if __name__ == "__main__":

    #####################################################################################################################################
    # Clear the screen
    os.system("clear")

    #####################################################################################################################################
    # LOAD ANALYSIS FILES

    # Load personal idiolect
    number_of_idioms, idiolect = load_iodlect(
        analysis_config.idiolect_file_path,
        analysis_config.contractions_dict,
    )

    # Load custom labels
    if analysis_config.analysis_label_file_path:
        with open(
            analysis_config.analysis_label_file_path,
            "r",
            encoding="utf-8",
        ) as file:
            labels = [line.strip() for line in file]

        number_of_labels = len(labels)
    else:
        labels = ""
        number_of_labels = None

    # Load custom questions from CSV
    if analysis_config.analysis_questions_file_path:
        with open(
            analysis_config.analysis_questions_file_path,
            mode="r",
            encoding="utf-8",
            newline="",
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            all_questions = [dict(row) for row in reader]

        # Now you have a list of dicts
        questions = all_questions
        number_of_questions = len(questions)
    else:
        questions = []
        number_of_questions = 0

    # # Load custom questions
    # if analysis_config.analysis_questions_file_path:
    #     with open(
    #         analysis_config.analysis_questions_file_path,
    #         "r",
    #         encoding="utf-8",
    #     ) as file:
    #         all_questions = [line.strip() for line in file]

    #     # questions = [all_questions[0] if all_questions else None]
    #     questions = all_questions
    #     number_of_questions = len(questions)
    # else:
    #     questions = ""
    #     number_of_questions = None

    #####################################################################################################################################
    # LOAD MODEL CONFIGURATION DATA
    model_configs = analysis_config.model_configs

    print_models = []
    local_print_models = []
    server_print_models = []
    active_model_configs = []

    # Start processing model configuration data
    for current_model in model_configs:
        use_model = current_model.use_model
        model_host = current_model.model_host
        model_name = current_model.model_name
        model_type = current_model.model_type

        # Skip this model if it's not marked for use
        if use_model.strip().lower() != "yes":
            continue

        if use_model.strip().lower() == "yes":
            active_model_configs.append(current_model)

            if model_host == "local":

                #####################################################################################################################################
                @logger.catch
                def format_second_level_dict(d):
                    """
                    Formats a nested dictionary up to two levels deep into an indented string.

                    This helper function takes a dictionary and returns a formatted string representation,
                    displaying only the first and second levels of nested keys and values. If any value at
                    the second level is itself a dictionary (i.e., a third level exists), it formats the
                    third level values but does not recurse beyond that.

                    Args:
                        d (dict): A dictionary with nested structure, where values at the first level
                            may themselves be dictionaries (and potentially include a third level).

                    Returns:
                        str: A string with a readable, line-by-line, indented representation of the
                        nested dictionary structure.

                    Side Effects:
                        None

                    Notes:
                        - This function is intended for **pretty-printing** or logging purposes, not for
                        serialization or machine-readable output.
                        - The formatting follows this indentation pattern:
                            key:
                            sub_key: sub_value
                            sub_key:
                                sub_sub_key: sub_sub_value
                        - The function does **not** format values that are not dictionaries at the top level;
                        they are skipped entirely.
                        - Depth is fixed: it does not recurse beyond the third level.

                    Caveats:
                        - Keys and values are converted to strings using `str()`, which may not preserve
                        original types or formatting (e.g., for dates or custom objects).
                        - The order of keys is preserved only if the input dictionary is an `OrderedDict`
                        or Python 3.7+ where dicts maintain insertion order.
                        - Top-level keys with non-dict values are ignored entirely (i.e., not shown in output).

                    Raises:
                        None
                    """
                    if is_dataclass(d):
                        d = asdict(d)

                    formatted = []
                    for key, value in d.items():
                        if isinstance(value, dict):
                            formatted.append(f"{key}:")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict):
                                    formatted.append(f"  {sub_key}:")
                                    for sub_sub_key, sub_sub_value in sub_value.items():
                                        formatted.append(
                                            f"    {sub_sub_key}: {sub_sub_value}"
                                        )
                                else:
                                    formatted.append(f"  {sub_key}: {sub_value}")
                        else:
                            formatted.append(f"{key}: {value}")
                    return "\n".join(formatted)

                #####################################################################################################################################

                second_level_info = format_second_level_dict(current_model)

                if second_level_info:
                    model_info = f"{model_name} / {model_type}\n{second_level_info}"
                else:
                    model_info = f"{model_name} / {model_type}"

                local_print_models.append(model_info)

            elif model_host == "server":
                server_details = []
                model_dict = asdict(current_model)  # Convert ModelConfig to dict
                for key, value in model_dict.items():
                    if key not in ["model_name", "model_type", "model_host"]:
                        if isinstance(value, dict):
                            server_details.append(f"\t{key}:")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict):
                                    server_details.append(f"\t\t{sub_key}:")
                                    for sub_sub_key, sub_sub_value in sub_value.items():
                                        server_details.append(
                                            f"\t\t\t{sub_sub_key}: {sub_sub_value}"
                                        )
                                else:
                                    server_details.append(f"\t\t{sub_key}: {sub_value}")
                        else:
                            server_details.append(f"\t{key}: {value}")

                server_info = f"{model_name} / {model_type}\n" + "\n".join(
                    server_details
                )
                server_print_models.append(server_info)

    analysis_directories_to_process = analysis_config.analysis_directories_to_process
    analysis_directories_to_ignore = analysis_config.analysis_directories_to_ignore
    analysis_output_directory = analysis_config.analysis_output_directory
    analysis_source_audio_file_directory = (
        analysis_config.analysis_source_audio_file_directory
    )

    # Get a list of all files in the directory
    json_files_in_directory = [
        entry
        for directory in analysis_directories_to_process
        if directory.is_dir()
        for entry in directory.iterdir()
        if entry.is_file() and str(entry).strip().lower().endswith(".json")
    ]
    audio_files_in_directory = [
        entry
        for entry in analysis_source_audio_file_directory.iterdir()
        if (entry.is_file() and str(entry).strip().lower().endswith((".mp3", ".flac")))
    ]

    # Create chunks
    for json_file_path in json_files_in_directory:
        # Obtain corresponding audio file
        audio_file_name = [
            item
            for item in audio_files_in_directory
            if json_file_path.stem.replace(" - large-v2 - SR", "") in item.name
        ]
    #####################################################################################################################################
    config_for_table = {
        key: value
        for key, value in {
            "analysis_directories_to_process": analysis_directories_to_process,
            "analysis_directories_to_ignore": analysis_directories_to_ignore,
            "analysis_output_directory": analysis_output_directory,
            "analysis_source_audio_file_directory": analysis_source_audio_file_directory,
            "idiolect_file_path": analysis_config.idiolect_file_path,
            "number_of_idioms": number_of_idioms,
            "analysis_label_file_path": analysis_config.analysis_label_file_path,
            "number_of_labels": number_of_labels,
            "analysis_questions_file_path": analysis_config.analysis_questions_file_path,
            "number_of_questions": number_of_questions,
            "contractions_dict": analysis_config.contractions_dict,
            "model_configs": print_models,
            "local_models": local_print_models,
            "server_models": server_print_models,
        }.items()
        if value  # This excludes None, empty strings, 0, False, [], {}, etc.
    }

    print_table(
        str(analysis_config.config_path),
        config_for_table,
        console,
        file_console,
        analysis_config.log_file,
    )

    # #####################################################################################################################################
    # Load all active models into memory
    dict_of_active_models = [asdict(model) for model in active_model_configs]
    logger.info("Loading AI models")
    # logger.info(
    #     f"active_model_configs:\n\n{json.dumps(dict_of_active_models, indent=4)}"
    # )
    active_models = load_models(dict_of_active_models)
    # logger.info(f"active_models:\n\n{json.dumps(active_models, indent=4)}")
    #####################################################################################################################################

# Regex to detect pattern: YYYY-MM-DD - HH-MM-SS before .json
skip_pattern = re.compile(
    r".*_\d{4}-\d{2}-\d{2} - \d{2}-\d{2}-\d{2}\.json$", re.IGNORECASE
)

# Ensure these are lists of Path objects if not already
analysis_directories_to_process = [Path(d) for d in analysis_directories_to_process]
analysis_directories_to_ignore = set(Path(d) for d in analysis_directories_to_ignore)


# Calculate total number of JSON files (non-recursive), skipping pattern matches
total_files = sum(
    len(
        [
            f
            for f in directory.iterdir()
            if f.is_file()
            and f.suffix.lower() == ".json"
            and not skip_pattern.match(f.name)
        ]
    )
    for directory in analysis_directories_to_process
    if directory not in analysis_directories_to_ignore
)

with tqdm(
    total=total_files, position=0, desc="Processing Files", unit="file"
) as file_pbar:
    for directory in analysis_directories_to_process:
        if directory in analysis_directories_to_ignore:
            continue

        json_files = [
            f
            for f in directory.iterdir()
            if f.is_file()
            and f.suffix.lower() == ".json"
            and not skip_pattern.match(f.name)
        ]

        for file_path in json_files:
            file_name = file_path.name
            logger.info(f"Processing: {file_path}")

            # Obtain corresponding audio file
            audio_file_name = [
                item
                for item in audio_files_in_directory
                if json_file_path.stem.replace(" - large-v2 - SR", "") in item.name
            ]

            # Your processing code here

            file_pbar.update(1)

            # Obtain date, time, and audio duration information from base filename
            (
                file_calendar_start_datetime,
                file_calendar_start_datetime_str,
                file_calendar_end_datetime,
                file_calendar_start_datetime_str,
                file_calendar_start_date,
                file_calendar_start_time,
                file_calendar_end_date,
                file_calendar_end_time,
                file_total_duration_in_seconds,
            ) = extract_date_time_from_json_filename(file_path)

            # Convert total duration from seconds to hours, minutes, seconds
            file_duration_hours, remainder = divmod(
                int(file_total_duration_in_seconds), 3600
            )
            file_duration_minutes, file_duration_seconds = divmod(remainder, 60)
            tmp_file_chunk_tags = []
            tmp_file_keyphrases = []
            #####################################################################################################################################

            # Create chunks
            chunks = create_chunk_data(file_path)

            all_chunk_details = []
            chunk_details = {}

            chunk_pbar = tqdm(total=len(chunks), position=1, unit="chunk", leave=False)

            # Start processing each chunk
            for chunk in chunks:
                chunk_calendar_start_datetime = chunk["transcription_time_data"][
                    "chunk_calendar_start_datetime"
                ]
                chunk_calendar_end_datetime = chunk["transcription_time_data"][
                    "chunk_calendar_end_datetime"
                ]
                chunk_text = chunk["chunk_text"]
                #####################################################################################################################################
                ai_model_results = generate_ai_model_results(
                    chunk_text,
                    active_model_configs,
                    active_models,
                    labels,
                    questions,
                    idiolect,
                    chunk_calendar_start_datetime,
                )
                #####################################################################################################################################

                chunk_transits = calculate_chunk_transits(
                    chunk_calendar_start_datetime,
                    analysis_config.astrology_variables.planet_and_aspect_orb,
                )

                chunk_profections = calculate_current_profections(
                    chunk_calendar_start_datetime, analysis_config.astrology_variables
                )

                chunk_zrs_data = generate_zrs_data(
                    analysis_config.astrology_variables.pos_file,
                    analysis_config.astrology_variables.pof_file,
                    chunk_calendar_start_datetime,
                )
                #####################################################################################################################################
                # Update the chunk_tags list with a sorted, unique list of tags from the QnA responses
                updated_chunk_tags = generate_chunk_tags(
                    chunk,
                    ai_model_results,
                    model_name="knowledgator/gliner-multitask-large-v0.5",
                    model_key="qna",
                    chunk_key="chunk_tags",
                )
                chunk["chunk_tags"] = updated_chunk_tags
                tmp_file_chunk_tags.append(updated_chunk_tags)
                #####################################################################################################################################
                # Update the chunk_keyphrases with keyphrases returned by the model
                updated_chunk_keyphrases = generate_chunk_tags(
                    chunk,
                    ai_model_results,
                    model_name="ml6team/keyphrase-extraction-kbir-inspec",
                    model_key=None,
                    chunk_key="chunk_keyphrases",
                )
                # print(updated_chunk_keyphrases)
                # input("\n\nHERE\n\n")
                chunk["chunk_keyphrases"] = updated_chunk_keyphrases
                tmp_file_keyphrases.append(updated_chunk_keyphrases)
                ####################################################################################################################################

                ####################################################################################################################################

                # Adjust the time we will use as the start time for the chunk within the audio file
                adjusted_chunk_audio_start_time_location = (
                    float(
                        chunk["transcription_time_data"][
                            "chunk_audio_start_time_location"
                        ]
                    )
                    - 1
                )

                # Adjust the time we will use as the end time for the chunk within the audio file
                adjusted_chunk_audio_end_time_location = (
                    float(
                        chunk["transcription_time_data"][
                            "chunk_audio_end_time_location"
                        ]
                    )
                    + 1
                )

                # This is most likely the first chunk, and if the start time is less than 1 second, we might
                # as well start from the very beginning of the audio file.
                if adjusted_chunk_audio_start_time_location < 1:
                    adjusted_chunk_audio_start_time_location = 0.00

                # Calculate the adjusted chunk duration
                # We will use this in the filname for the chunk audio file
                chunk_duration = int(
                    adjusted_chunk_audio_end_time_location
                    - adjusted_chunk_audio_start_time_location
                )

                chunk_audio_file_name = f"{audio_file_name[0].stem} - chunk - {chunk["chunk_id"]:0{4}} of {len(chunks):0{4}} - {chunk_duration}{audio_file_name[0].suffix}"
                chunk_json_file_name = f"{audio_file_name[0].stem} - chunk - {chunk["chunk_id"]:0{4}} of {len(chunks):0{4}} - {chunk_duration}.json"

                chunk_source_file_data = {
                    "source_audio_file_name": audio_file_name[0].name,
                    "source_json_file_name": json_file_path.name,
                    "chunk_audio_file_name": chunk_audio_file_name,
                    "total_number_of_chunks": len(chunks),
                }

                updated_chunk = insert_keys(
                    chunk, insert_after_key="chunk_id", new_items=chunk_source_file_data
                )

                # Parse string to datetime
                chunk_calendar_start_datetime_dt = datetime.fromisoformat(
                    chunk_calendar_start_datetime
                )
                chunk_calendar_end_datetime_dt = datetime.fromisoformat(
                    chunk_calendar_end_datetime
                )

                ####################################################################################################################################
                # Adjust times
                adjusted_chunk_calendar_start_datetime = (
                    chunk_calendar_start_datetime_dt + timedelta(seconds=-1)
                )
                adjusted_chunk_calendar_end_datetime = (
                    chunk_calendar_end_datetime_dt + timedelta(seconds=1)
                )

                adjusted_chunk_duration = calculate_duration(
                    adjusted_chunk_calendar_start_datetime,
                    adjusted_chunk_calendar_end_datetime,
                    key_prefix="adjusted",
                )

                adjusted_chunk_time_data = {
                    "adjusted_chunk_time_data": {
                        "adjusted_chunk_audio_start_time_location": adjusted_chunk_audio_start_time_location,
                        "adjusted_chunk_calendar_start_datetime": adjusted_chunk_calendar_start_datetime.isoformat(),
                        "adjusted_chunk_audio_end_time_location": adjusted_chunk_audio_end_time_location,
                        "adjusted_chunk_calendar_end_datetime": adjusted_chunk_calendar_end_datetime.isoformat(),
                        **adjusted_chunk_duration,
                    }
                }

                updated_chunk = insert_keys(
                    updated_chunk,
                    insert_after_key="transcription_time_data",
                    new_items=adjusted_chunk_time_data,
                )
                ####################################################################################################################################

                ####################################################################################################################################
                # The dictionary that contains all of the data regarding the analysis of a chunk
                chunk_details = {
                    # **chunk,
                    **updated_chunk,
                    "chunk_analysis": {
                        **ai_model_results,
                        **chunk_transits,
                        **chunk_profections,
                        **chunk_zrs_data,
                    },
                }
                #####################################################################################################################################
                all_chunk_details.append(chunk_details)

                chunk_pbar.close()
                #####################################################################################################################################
            file_pbar.update()
            #####################################################################################################################################

            # Flatten the list of lists and deduplicate
            file_all_chunk_tags = sorted(
                set(tag for sublist in tmp_file_chunk_tags for tag in sublist)
            )

            file_all_keyphrases = sorted(
                set(
                    keyphrase
                    for sublist in tmp_file_keyphrases
                    for keyphrase in sublist
                )
            )

            file_chunk_root = generate_chunk_root(all_chunk_details)

            file_time_data = {
                "file_time_data": {
                    "file_calendar_start_datetime": file_calendar_start_datetime,
                    "file_calendar_start_date": file_calendar_start_date,
                    "file_calendar_start_time": file_calendar_start_time,
                    "file_calendar_end_datetime": file_calendar_start_datetime,
                    "file_calendar_end_date": file_calendar_end_date,
                    "file_calendar_end_time": file_calendar_end_time,
                    "file_duration_hours": file_duration_hours,
                    "file_duration_minutes": file_duration_minutes,
                    "file_duration_seconds": file_duration_seconds,
                    "file_total_duration_in_seconds": file_total_duration_in_seconds,
                }
            }

            audio_file_metadata, original_meta_data = generate_audio_metadata(audio_file_name[0])

            # The dictionary that makes up the metadata file for each audio journal analusr
            file_contents = {
                "original_trasncript_filename": file_name,
                # "source_audio_file_name": audio_file_name[0].name,
                **audio_file_metadata,
                **file_time_data,
                "file_all_chunk_tags": file_all_chunk_tags,
                "file_all_keyphrases": file_all_keyphrases,
                "file_chunk_root": file_chunk_root,
                "chunks": all_chunk_details,
            }
            #####################################################################################################################################
            # Adding a time stamp suffix for texting purposes; this will need to be removed in production
            file_name_suffix = datetime.now().strftime("%Y-%m-%d - %H-%M-%S")

            # Construct the full output file path and file name
            output_file_path = (
                analysis_output_directory
                / f"{Path(file_name).stem} - analysis_{file_name_suffix}.json"
            )

            # Ensure the folder exists
            os.makedirs(analysis_output_directory, exist_ok=True)

            # Write the file_content dictionary to as a JSON object
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(json.dumps(file_contents, indent=4))

            logger.info(f"Analysis written to: {output_file_path}\n")
            #####################################################################################################################################
        file_pbar.close()
#####################################################################################################################################
