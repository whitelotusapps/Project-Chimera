"""
THIS FILE TRANSCRIBED THE AUDIO FILE AND NAMES THE MP3 ACCORDING TO THE BELOW FORMAT:

<START DATE> - <START TIME> - <END DATE> - <END TIME> - <TOTAL DURATION OF MP3 IN SECONDS> - <REMAINING FILENAME OF THE MP3>.MP3

THEN THE TRANSCRIBED FILES WILL MATCH THE RENAMED MP3 FORMAT ABOVE.
"""

import datetime

#####################################################################################################################################
# NATIVE MODULES
#####################################################################################################################################
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

#####################################################################################################################################
# HELPER MODULES
#####################################################################################################################################
from loguru import logger
from rich.console import Console

from modules.date_functions import (
    get_audio_file_datetime_from_system,
    get_start_date_and_end_date,
    is_past_date,
)
from modules.helper_functions import convert_to_mp3, print_table, replace_text
from modules.project_paths import PATHS
from modules.transcript_config_loader import load_transcription_config
from modules.transcript_file_name_functions import (
    first_part_of_filename,
    update_filename_and_get_new,
)
from modules.transcript_tags_and_art import populate_flac_tags, populate_mp3_tags

#####################################################################################################################################

#####################################################################################################################################
# Obtain the realtive path of where the script is being ran from
#####################################################################################################################################
if getattr(sys, "frozen", False):
    # Running as a single-file executable
    script_path = Path(sys.executable).parent.resolve()
else:
    # Running as a script
    script_path = Path(__file__).parent
#####################################################################################################################################

#####################################################################################################################################
# CONFIG FILE CHECK
#####################################################################################################################################
config = load_transcription_config()

#####################################################################################################################################
# Create a rich console
console = Console()

# Create a console for printing the config table to our logfile
file_console = Console(color_system=None)

# This is use to ensure that we print the config table only one time
print_table_check = True

#####################################################################################################################################

#####################################################################################################################################
# Function to process files in a directory and its subdirectories
#####################################################################################################################################
@logger.catch
def process_files_in_directory(
    files,
    print_table_check,
    transcription_output_path,
    search_and_replace_pairs,
    search_and_replace_transcription_output_path,
    save_to_gpt4all_localdocs,
    gpt4all_localdocs_path,
    currently_transcribed_files,
    desired_date,
):
    """
    Process a list of files in a directory for transcription and subsequent processing.

    Args:
        files (list of str): List of filenames to process.
        print_table_check (bool): Flag to determine if config table should be printed once.
        transcription_output_path (str): Directory path to save transcription outputs.
        search_and_replace_pairs (list of dict): List of search and replace dictionaries for text substitution.
        search_and_replace_transcription_output_path (str): Directory path to save search-and-replace processed transcriptions.
        save_to_gpt4all_localdocs (str): Flag ("yes" or other) indicating if files should be saved to GPT4ALL LocalDocs folder.
        gpt4all_localdocs_path (str): Path to the GPT4ALL LocalDocs folder.
        currently_transcribed_files (list of str): List of files already transcribed to avoid duplicate processing.
        desired_date (datetime): Minimum file datetime threshold for processing.

    Side Effects:
        - Converts non-MP3 audio files to MP3 format when necessary.
        - Renames and reformats MP3 filenames based on timestamps.
        - Performs transcription by invoking an external tool via system call.
        - Applies search-and-replace text transformations to transcription outputs.
        - Saves processed transcription files to designated directories.
        - Optionally copies transcriptions to a GPT4ALL local documents directory.
        - Embeds metadata and cover art into MP3 files.
        - Creates directories if they do not exist.
        - Logs extensive information about processing steps, errors, and status.

    Notes:
        - The function depends on various external utilities and global variables such as
        `file_extensions_to_transcribe`, `transcription_model`, `faster_whisper_binary_path`,
        and logging configuration.
        - Error handling for transcription and file operations logs errors and continues processing other files.
        - The processing order is chronological based on file timestamps.
        - Search-and-replace pairs should be properly formed with valid regex or string patterns.
        - The function assumes environment setup with necessary binaries and permissions.
        - The transcription process uses an external executable invoked via `os.system`; security considerations apply.
        - Large numbers of files or large file sizes may affect performance and resource usage.
        - The parameter `print_table_check` controls whether configuration tables are printed only once.

    Caveats:
        - Files with unexpected or invalid naming conventions might be skipped or misprocessed.
        - File locks, permissions, or concurrent access issues could cause failures.
        - This function has multiple responsibilities (conversion, transcription, file management, tagging)
        which might be better refactored into smaller, testable components.
    """

    files_to_process = []

    for file_path in files:
        # Ensure file_path is a Path object (should already be)
        # Get the datetime from the file (pass Path or str as needed)
        file_datetime = get_audio_file_datetime_from_system(
            file_path
        )  # if this function expects str, do str(file_path)

        file_extension = file_path.suffix.lower()

        if file_extension not in config.file_extensions_to_transcribe:
            logger.info(f"SKIPPING FILE: {str(file_path)}")
            continue

        # For non-MP3 and non-FLAC files, convert and check for existing transcription
        if file_extension not in config.corpus_extensions:
            transcript_file_starts_with = first_part_of_filename(
                file_path
            )  # should accept Path or str

            # currently_transcribed_files presumably list of filenames (str), so convert Path to str if needed
            transcript_exists = any(
                transcript_file_starts_with in f for f in currently_transcribed_files
            )

            if not transcript_exists:
                logger.info(
                    f"{str(file_path)} is not a corpus accepted file, converting to MP3"
                )
                new_file_path = convert_to_mp3(file_path)
                if new_file_path is None:
                    continue

                file_path = new_file_path
                file_extension = (
                    file_path.suffix.lower()
                )  # Use Path method instead of os.path.splitext

        if file_extension in config.corpus_extensions:
            if (
                file_path.stat().st_size > 0
                and datetime.fromtimestamp(file_datetime) >= desired_date
            ):
                new_file_path = update_filename_and_get_new(
                    file_path, config.corpus_extensions_pattern
                )

                if new_file_path:
                    base_name = Path(new_file_path).stem

                if not file_path:
                    continue
            else:
                base_name = Path(file_path).stem

            # logger.info(f"new_file_path.name: {new_file_path.name}")
            # input("\n\nHERE - LAST ONE\n\n")
            start_date, start_time, end_date, end_time, comment, extension = (
                get_start_date_and_end_date(
                    new_file_path.name, config.corpus_extensions_pattern
                )
            )
            if is_past_date(start_date):
                files_to_process.append(new_file_path)

    files_to_transcribe = []
    for file_to_process in files_to_process:
        base_name = Path(file_to_process).stem

        transcript_file = (
            search_and_replace_transcription_output_path
            / "JSON"
            / f"{base_name} - {config.transcription_model} - SR.json"
        )

        if not transcript_file.exists():
            files_to_transcribe.append(file_to_process)

    files_to_transcribe = sorted(files_to_transcribe)
    logger.info(
        "files_to_transcribe:\n\n"
        + "\n".join(f"\t{p}" for p in files_to_transcribe)
        + "\n"
    )

    for file_to_transcribe in files_to_transcribe:
        base_name = Path(file_to_transcribe).stem

        transcript_file = (
            search_and_replace_transcription_output_path
            / "JSON"
            / f"{base_name} - {config.transcription_model} - SR.json"
        )

        if print_table_check:
            # Build the log file path
            logfile_name = datetime.now().strftime(
                "%Y-%m-%d - %H-%M-%S - transcriptions.log"
            )
            logfile_path_with_name = PATHS.logs / logfile_name

            # Add the log file to Loguru
            logger.add(logfile_path_with_name)

            # Call print_table, converting the Path to a string
            print_table(
                str(config.config_path),
                config.transcription_variables,
                console,
                file_console,
                config.log_file,
            )
            print_table_check = False

        logger.opt(colors=True).info(
            f"<white><b>Processing file:</b></white> <green>{file_to_transcribe}</green>"
        )
        # Perform transcription
        os.system(
            f'{config.faster_whisper_binary_path} "{file_to_transcribe}" --language=en --model_dir="{config.faster_whisper_model_path}" --model={config.transcription_model} --task transcribe --print_progress --output_format all --beep_off --output_dir "{config.transcription_output_path}"'
        )

        # List all files in the transcription output directory
        transcription_output_files = [
            f.name
            for f in transcription_output_path.iterdir()
            if f.is_file() and f.name.startswith(base_name + ".")
        ]

        for output_file in transcription_output_files:
            file_extension = output_file.split(".")[-1].upper()
            destination_directory = transcription_output_path / file_extension

            if not os.path.exists(destination_directory):
                logger.opt(colors=True).info(
                    f"<red><b>Creating</b></red> <green>{destination_directory}</green>"
                )
                destination_directory.mkdir(parents=True, exist_ok=True)

            filename_with_model_name = (
                f"{base_name} - {config.transcription_model}.{file_extension.lower()}"
            )
            search_and_replace_name = f"{base_name} - {config.transcription_model} - SR.{file_extension.lower()}"
            transcription_output_file_path = transcription_output_path / output_file

            logger.opt(colors=True).info(
                f"<yellow><b>Performing search and replace on:</b></yellow> <green>{transcription_output_file_path}</green>"
            )

            try:
                # Read the original file content
                with open(transcription_output_file_path, encoding="utf-8") as f:
                    file_content = f.read()

                # Apply the replace_text function
                new_file_content = replace_text(file_content, search_and_replace_pairs)

                # Write the modified content to the new file in the search and replace directory
                search_and_replace_file_path = os.path.join(
                    search_and_replace_transcription_output_path,
                    file_extension,
                    search_and_replace_name,
                )

                if not (
                    search_and_replace_transcription_output_path / file_extension
                ).exists():

                    logger.opt(colors=True).info(
                        f"<red><b>Creating</b></red> <green>{search_and_replace_transcription_output_path}</green>"
                    )

                    (
                        search_and_replace_transcription_output_path / file_extension
                    ).mkdir(parents=True, exist_ok=True)

                # Search and replace directory location
                with open(search_and_replace_file_path, "w", encoding="utf-8") as f:
                    f.write(new_file_content)
                    logger.opt(colors=True).info(
                        f"<white><b>Saved</b></white> <cyan>{search_and_replace_name}</cyan> <white>to</white> <green>{search_and_replace_file_path}</green>"
                    )

                if save_to_gpt4all_localdocs.lower() == "yes":
                    # Save transcribed .text, and search and replaced, files to the GPT4ALL LocalDocs folder location as .txt files
                    if (
                        output_file.lower().endswith(".text")
                        and len(new_file_content) > 0
                    ):  # Corrected file extension
                        gpt4all_file_path_with_name = os.path.join(
                            gpt4all_localdocs_path, search_and_replace_name
                        )

                        with open(
                            gpt4all_file_path_with_name, "w", encoding="utf-8"
                        ) as f:
                            f.write(new_file_content)
                            logger.opt(colors=True).info(
                                f"<GREEN><white><b>GPT4ALL file</b></white></GREEN> <yellow>{search_and_replace_name}</yellow> <white>saved to</white> <green>{gpt4all_file_path_with_name}</green>"
                            )
                else:
                    logger.opt(colors=True).info(
                        f"<RED><white><b>save_to_gpt4all_localdocs is set to:</b></white></RED> <yellow><b>{save_to_gpt4all_localdocs}</b></yellow>"
                    )

                # Move the original file to the corresponding subfolder
                shutil.move(
                    transcription_output_path / output_file,
                    destination_directory / filename_with_model_name,
                )

                logger.opt(colors=True).info(
                    f"<white>Moved</white> <cyan>'{output_file}'</cyan> <white>to</white> <magenta>'{destination_directory}'</magenta> <white>as</white> <green>'{filename_with_model_name}</green>"
                )

            except Exception as e:
                logger.opt(colors=True).error(
                    f"<RED><white><b>Error opening file</b></white></RED> <yellow><b>{output_file}:</b></yellow> {e}"
                )

        #####################################################################################################################################

        txt_path = (
            Path(search_and_replace_transcription_output_path)
            / "TXT"
            / f"{base_name} - {config.transcription_model} - SR.txt"
        )
        text_path = (
            Path(search_and_replace_transcription_output_path)
            / "TEXT"
            / f"{base_name} - {config.transcription_model} - SR.text"
        )

        if file_to_transcribe.suffix.lower() == ".mp3":
            logger.info("Embedding MP3 tags and updating cover art...")
            populate_mp3_tags(
                file_to_transcribe,
                files_to_transcribe,
                txt_path,
                text_path,
                "The Real Zack Olinger",
                config.word_cloud.top_n_words,
                config.word_cloud.custom_stopwords,
                config.corpus_extensions_pattern,
            )

        if file_to_transcribe.suffix.lower() == ".flac":
            logger.info("Embedding FLAC tags and updating cover art...")
            populate_flac_tags(
                file_to_transcribe,
                files_to_transcribe,
                txt_path,
                text_path,
                "The Real Zack Olinger",
                config.word_cloud.top_n_words,
                config.word_cloud.custom_stopwords,
                config.corpus_extensions_pattern,
            )


#####################################################################################################################################

#####################################################################################################################################
# Process files in each directory and its subdirectories
#####################################################################################################################################

for directory in config.audio_file_directories_to_process:
    dir_path = Path(directory).resolve()

    # Skip if directory is in ignore list (or inside one)
    if any(
        dir_path.is_relative_to(ignore_path.resolve())
        for ignore_path in config.audio_file_directories_to_ignore
    ):
        continue

    # Get files only (no recursion)
    files = [p for p in dir_path.iterdir() if p.is_file()]

    # Pass files list to your processing function
    process_files_in_directory(
        files,
        print_table_check,
        config.transcription_output_path,
        config.search_and_replace_pairs,
        config.search_and_replace_transcription_output_path,
        config.save_to_gpt4all_localdocs,
        config.gpt4all_localdocs_path,
        config.currently_transcribed_files,
        config.desired_date,
    )


#####################################################################################################################################
