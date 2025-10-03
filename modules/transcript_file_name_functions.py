import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger
from mutagen.flac import FLAC
from mutagen.mp3 import MP3

from .date_functions import get_audio_file_datetime_from_system


#####################################################################################################################################
@logger.catch
def first_part_of_filename(input_file):
    """
    Generate a formatted date-time string based on the file's creation or modification timestamp.

    This function retrieves the file's creation date (or modification date if creation is unavailable)
    and returns it formatted as a string in the pattern "YYYY-MM-DD - HH-MM-SS".

    Args:
        input_file (str): Path to the input file.

    Returns:
        str: Formatted date-time string representing the file's creation or last modification time.

    Side Effects:
        None.

    Notes:
        - Relies on `get_file_datetime()` to obtain the file timestamp.
        - The timestamp is converted to a datetime object and formatted using strftime.
        - The output format is suitable for use in filenames or logs.

    Caveats:
        - If the file does not exist or is inaccessible, an exception will be raised by `get_file_datetime` or `datetime.fromtimestamp`.
        - Timezone information is not included; the output reflects local system time.

    Example:
        >>> first_part_of_filename("example.mp3")
        '2025-07-02 - 14-30-15'
    """
    start_datetime = datetime.fromtimestamp(
        get_audio_file_datetime_from_system(input_file)
    )
    return start_datetime.strftime("%Y-%m-%d - %H-%M-%S")


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def filename_matches_format(filename, file_extensions_to_transcribe_pattern):
    """
    Check if a filename matches the expected audio transcription file naming format.

    The expected format is:
    YYYY-MM-DD - HH-MM-SS - YYYY-MM-DD - HH-MM-SS - <number> - <comment>.<extension>
    where <extension> matches one of the allowed file extensions.

    Args:
        filename (str): The filename to validate.

    Returns:
        bool: True if the filename matches the pattern, False otherwise.

    Side Effects:
        None.

    Notes:
        - Uses a compiled regular expression with `re.IGNORECASE` for case-insensitive matching.
        - The pattern for allowed file extensions is dynamically inserted from the variable
          `file_extensions_to_transcribe_pattern`.
        - The function assumes `file_extensions_to_transcribe_pattern` is defined and
          formatted as a regex pattern string representing file extensions (e.g., "mp3|wav|flac").
        - Matches strictly the entire filename (from start to end).
        - Useful for filtering or validating filenames before processing.

    Caveats:
        - If `file_extensions_to_transcribe_pattern` is not defined or incorrectly formatted,
          this function will raise a NameError or behave unexpectedly.
        - This function does not check the validity of date/time values, only the string format.
    """
    # logger.info(
    #     f"file_extensions_to_transcribe_pattern: {file_extensions_to_transcribe_pattern}"
    # )
    pattern = re.compile(
        r"^\d{{4}}-\d{{2}}-\d{{2}} - \d{{2}}-\d{{2}}-\d{{2}} - \d{{4}}-\d{{2}}-\d{{2}} - \d{{2}}-\d{{2}}-\d{{2}} - \d+ - .+\.({})$".format(
            file_extensions_to_transcribe_pattern
        ),
        re.IGNORECASE,
    )

    # logger.info(f"filename: {filename}")
    # logger.info(f"{bool(pattern.match(filename))}")
    # input("\n\nHERE\n\n")
    return bool(pattern.match(filename))


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
# def update_filename_and_get_new(file_path, file_extension_pattern):
def update_filename_and_get_new(
    file_path: Path, file_extension_pattern: str
) -> Optional[Path]:
    """
    Ensure a file's name matches the expected naming convention; rename it if necessary.

    The expected filename format is:
    YYYY-MM-DD - HH-MM-SS - YYYY-MM-DD - HH-MM-SS - <duration_seconds> - <rest_of_filename>

    If the current filename does not match the format but contains a valid
    start datetime and additional parts, this function calculates the end datetime
    based on the audio duration and renames the file accordingly.

    Args:
        file_path (str): The full path to the audio file to check and possibly rename.

    Returns:
        str or None: The updated full file path if renamed or already correctly named;
                     None if the filename is invalid or renaming failed.

    Side Effects:
        - Renames the file on disk if the filename does not match the expected format.
        - Logs information and errors about the renaming process.

    Notes:
        - Uses `filename_matches_format()` to validate the filename format.
        - Assumes the filename starts with at least two parts representing start date and time,
          separated by ' - '.
        - Uses `mutagen.mp3.MP3` to read audio metadata and get duration in seconds.
        - Constructs new filename by appending calculated end datetime and duration.
        - Preserves the rest of the original filename parts after the start datetime.

    Caveats:
        - If the filename does not have at least 3 parts separated by ' - ', logs an error and returns None.
        - The function expects valid datetime strings in the first two parts of the filename.
        - If the audio file is invalid or unreadable by mutagen, this may raise an exception.
        - Renaming may fail due to file permission errors or if the target filename already exists.
        - Does not handle such exceptions internally; consider wrapping calls in try-except if needed.
    """
    # logger.info("")
    filename = file_path.name
    file_extension = file_path.suffix.lower()

    # logger.info(f"filename: {filename}")
    # logger.info(f"file_extension: {file_extension}")  # .mp3
    # logger.info(
    #     f"file_extension_pattern: {file_extension_pattern}"
    # )  # ".mp3|.ogg|.flac"
    # input("\n\nHERE\n\n")

    # Validate file naming pattern
    if filename_matches_format(filename, file_extension_pattern):
        return file_path
    else:
        parts = filename.split(" - ")
        if len(parts) < 3:
            logger.error(f"{file_path} does not match naming conventions.")
            return None

        # Parse datetime from filename
        try:
            start_datetime = datetime.strptime(
                f"{parts[0]} {parts[1]}", "%Y-%m-%d %H-%M-%S"
            )
        except ValueError as e:
            logger.error(f"Date parsing failed for {file_path}: {e}")
            return None

        # Read audio duration
        try:
            if file_extension == ".mp3":
                logger.info(f"Processing MP3 file: {file_path}")
                audio = MP3(file_path)
            elif file_extension == ".flac":
                logger.info(f"Processing FLAC file: {file_path}")
                audio = FLAC(file_path)
            else:
                logger.error(f"Unsupported audio format for {file_path}")
                return None

            duration = int(audio.info.length)
        except Exception as e:
            logger.error(f"Error reading audio metadata for {file_path}: {e}")
            return None

        end_datetime = start_datetime + timedelta(seconds=duration)

        # Build new filename
        new_filename = (
            f"{start_datetime.strftime('%Y-%m-%d - %H-%M-%S')} - "
            f"{end_datetime.strftime('%Y-%m-%d - %H-%M-%S')} - "
            f"{duration} - {' - '.join(parts[2:])}"
        )
        new_file_path = file_path.with_name(new_filename)

        # Rename the file
        try:
            file_path.rename(new_file_path)
            logger.opt(colors=True).info(
                f"<white><b>Renamed</b></white> <cyan>'{file_path}'</cyan> <white>to</white> <green>'{new_file_path}'</green>"
            )
            return new_file_path
        except Exception as e:
            logger.error(f"Failed to rename {file_path} to {new_file_path}: {e}")
            return None


#####################################################################################################################################
