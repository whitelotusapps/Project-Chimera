import datetime
import platform
import re
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


#####################################################################################################################################
@logger.catch
def dates_outside_24_hours(date1: str, date2: str) -> bool:
    """
    Determine if both given dates fall outside a ±24-hour window from the current time.

    This function parses two ISO-formatted date strings (YYYY-MM-DD) and compares them
    against the current datetime to check whether both dates are either older than 24 hours
    before now or more than 24 hours after now.

    Args:
        date1 (str): The first date string in ISO format ("YYYY-MM-DD").
        date2 (str): The second date string in ISO format ("YYYY-MM-DD").

    Returns:
        bool: True if both dates are either before the past 24-hour cutoff or after the future
        24-hour cutoff relative to the current time; False otherwise.

    Side Effects:
        - None (pure function with no mutations or external interactions).

    Notes:
        - The input date strings are assumed to represent dates without time (midnight).
        - Time components are defaulted to midnight when parsing.
        - The function uses the system's current local datetime (`datetime.now()`).
        - Dates exactly within the 24-hour window (between past_24h and future_24h) will return False.
        - The function returns True only if both dates lie strictly outside the 24-hour window in the same direction.

    Caveats:
        - The function does not handle timezone-aware datetime objects.
        - Passing improperly formatted date strings will raise a `ValueError`.
        - This function compares only dates, ignoring hours, minutes, and seconds of input dates beyond midnight.

    Raises:
        ValueError: If either `date1` or `date2` does not match the "%Y-%m-%d" format.
    """
    now = datetime.now()
    past_24h = now - timedelta(hours=24)
    future_24h = now + timedelta(hours=24)

    dt1 = datetime.strptime(date1, "%Y-%m-%d")
    dt2 = datetime.strptime(date2, "%Y-%m-%d")

    return dt1 < past_24h and dt2 < past_24h or dt1 > future_24h and dt2 > future_24h


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def is_past_date(start_date: str) -> bool:
    """
    Check if the given date string represents a date before today.

    This function parses a date string in the "YYYY-MM-DD" format and compares it
    to the current local date to determine if it is in the past.

    Args:
        start_date (str): The date string to check, in ISO format ("YYYY-MM-DD").

    Returns:
        bool: True if the given date is earlier than today's date, False otherwise.

    Side Effects:
        - None. This is a pure function with no side effects.

    Notes:
        - Comparison is done using date objects only; time components are ignored.
        - Uses the system's local current date (`datetime.now().date()`).
        - If the date matches today's date, the function returns False.

    Caveats:
        - Input must strictly follow the "YYYY-MM-DD" format or a `ValueError` will be raised.
        - Does not handle timezone-aware dates or times.

    Raises:
        ValueError: If `start_date` is not in the expected format.
    """
    return datetime.strptime(start_date, "%Y-%m-%d").date() < datetime.now().date()


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def get_start_date_and_end_date(corpus_file, corpus_extensions_pattern):
    """
    Extract the start and end dates from an MP3 filename following a specific naming convention.

    The function expects the filename to follow the pattern:
    "YYYY-MM-DD - HH-MM-SS - YYYY-MM-DD - HH-MM-SS - <number> - <comment>.mp3"
    and extracts the first and third date components as the start and end dates respectively.

    Args:
        mp3_file (str): The MP3 filename to parse.

    Returns:
        tuple: A tuple containing two strings:
            - start_date (str): The start date in "YYYY-MM-DD" format.
            - end_date (str): The end date in "YYYY-MM-DD" format.

    Side Effects:
        - None. The function performs no I/O or state mutations.

    Notes:
        - The function uses regular expressions to parse the filename.
        - Only the dates (not times or other components) are returned.
        - Assumes the filename strictly matches the expected pattern.

    Caveats:
        - If the filename does not match the pattern, an `AttributeError` or `ValueError` will be raised
          because `match` will be `None` and `match.groups()` will fail.
        - Does not validate the extracted date strings beyond regex matching.

    Raises:
        AttributeError: If `mp3_file` does not match the expected pattern and `match` is None.
    """

    pattern = re.compile(
        rf"^(\d{{4}}-\d{{2}}-\d{{2}}) - (\d{{2}}-\d{{2}}-\d{{2}}) - (\d{{4}}-\d{{2}}-\d{{2}}) - (\d{{2}}-\d{{2}}-\d{{2}}) - \d+ - (.+)\.({corpus_extensions_pattern})$",
        re.IGNORECASE,
    )

    # logger.info(f"corpus_file.name: {corpus_file}")
    # logger.info(f"ext_patter: {corpus_extensions_pattern}")
    # logger.info(f"pattern: {pattern}")

    # input("\n\nHERE\n\n")
    match = pattern.match(corpus_file)
    start_date, start_time, end_date, end_time, comment, extension = match.groups()

    # logger.info(f"start_date: {start_date}")
    # logger.info(f"start_time: {start_time}")
    # logger.info(f"end_date: {end_date}")
    # logger.info(f"end_time: {end_time}")
    # logger.info(f"comment: {comment}")
    # logger.info(f"extension: {extension}")

    # input("\n\nHERE\n\n")

    return start_date, start_time, end_date, end_time, comment, f".{extension}"


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
# def get_file_datetime(path_to_file):
def get_audio_file_datetime_from_system(path_to_file: Path):
    """
    Retrieve the creation date of a file if available; otherwise, fallback to the last modification date.

    This function attempts to get the file's creation time on Windows and macOS. On Linux or systems
    where creation time is unavailable, it returns the last modification time instead.

    Args:
        path_to_file (str): Path to the file whose datetime is to be retrieved.

    Returns:
        float: Timestamp representing the file's creation time or last modification time, as seconds
               since the epoch (Unix timestamp).

    Side Effects:
        None.

    Notes:
        - On Windows, uses `os.path.getmtime` due to platform-specific behavior (modification time).
        - On Unix-like systems, attempts to access `st_birthtime` (creation time) if available.
        - On Linux, creation time is typically not available; falls back to `st_mtime` (last modification time).
        - Returned timestamp is suitable for passing to `datetime.fromtimestamp()` for conversion to datetime object.
        - Inspiration: http://stackoverflow.com/a/39501288/1709587

    Caveats:
        - The meaning of "creation time" may vary by file system and OS.
        - On some platforms, creation time might not be supported or reliable.
        - This function does not handle exceptions such as file not found; callers should handle them as needed.

    Example:
        >>> timestamp = get_file_datetime("/path/to/file.txt")
        >>> import datetime
        >>> datetime.datetime.fromtimestamp(timestamp)
        datetime.datetime(2023, 7, 2, 15, 30, 0)
    """

    path_to_file = Path(path_to_file)  # ensure it's a Path object

    if platform.system() == "Windows":
        return path_to_file.stat().st_mtime  # equivalent to os.path.getmtime
    else:
        stat = path_to_file.stat()
        try:
            return stat.st_birthtime
        except AttributeError:
            # On Linux, no creation time, so use last modification time instead
            return stat.st_mtime


#####################################################################################################################################


#####################################################################################################################################
def extract_date_time_from_json_filename(file_path):
    """
    Extracts datetime and duration information from a structured filename.

    The filename is expected to follow a specific pattern:
    'YYYY-MM-DD - HH-MM-SS - YYYY-MM-DD - HH-MM-SS - DURATION - [description].json'

    Args:
        file_path (str): Full path to the JSON file with a filename containing datetime metadata.

    Returns:
        tuple:
            - start_datetime (datetime): Combined start date and time as a datetime object.
            - end_datetime (datetime): Combined end date and time as a datetime object.
            - start_date (str): Start date in 'YYYY-MM-DD' format.
            - start_time (str): Start time in 'HH-MM-SS' format.
            - end_date (str): End date in 'YYYY-MM-DD' format.
            - end_time (str): End time in 'HH-MM-SS' format.
            - duration_in_seconds (str): Duration represented as a string of seconds.

    Raises:
        ValueError: If the filename does not conform to the expected pattern.

    Side Effects:
        - None.

    Notes:
        - Only filenames that strictly match the defined pattern will be processed.
        - The time segments in the filename use hyphens (e.g., 'HH-MM-SS') and are converted to colons.
        - The returned `duration_in_seconds` is not cast to int and remains a string for consistency with the original match.

    Caveats:
        - Will fail silently if the path is valid but the filename does not match the regex.
        - Assumes the filename includes full metadata in the expected structured format.
    """

    filename = Path(file_path).name  # Extract filename from the full path

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}) - (\d{2}-\d{2}-\d{2}) - "
        r"(\d{4}-\d{2}-\d{2}) - (\d{2}-\d{2}-\d{2}) - (\d+) - .+\.json$",
        re.IGNORECASE,
    )

    match = pattern.match(filename)
    if match:
        start_date, start_time, end_date, end_time, duration_in_seconds = match.groups()

        start_datetime_str = f"{start_date} {start_time}".replace("-", ":", 2).replace(
            "-", ":"
        )
        end_datetime_str = f"{end_date} {end_time}".replace("-", ":", 2).replace(
            "-", ":"
        )

        # Convert to datetime objects
        start_datetime = datetime.strptime(start_datetime_str, "%Y:%m:%d %H:%M:%S")
        end_datetime = datetime.strptime(end_datetime_str, "%Y:%m:%d %H:%M:%S")

        start_datetime_str = datetime.strptime(start_datetime_str, "%Y:%m:%d %H:%M:%S")
        end_datetime_str = datetime.strptime(end_datetime_str, "%Y:%m:%d %H:%M:%S")

        return (
            start_datetime.isoformat(),
            start_datetime_str,
            end_datetime.isoformat(),
            end_datetime_str,
            start_date,
            start_time.replace("-", ":"),
            end_date,
            end_time.replace("-", ":"),
            duration_in_seconds,
        )
    else:
        raise ValueError(f"Filename does not match the expected pattern: {filename}")

    # filename = ntpath.basename(file_path)  # Extract filename from the full path
    # pattern = re.compile(
    #     r"(\d{4}-\d{2}-\d{2}) - (\d{2}-\d{2}-\d{2}) - (\d{4}-\d{2}-\d{2}) - (\d{2}-\d{2}-\d{2}) - (\d+) - .+\.json$",
    #     re.IGNORECASE,
    # )
    # match = pattern.match(filename)
    # if match:
    #     start_date, start_time, end_date, end_time, duration_in_seconds = match.groups()

    #     start_datetime_str = f"{start_date} {start_time}".replace(
    #         "-", ":"
    #     )  # Adjust to match datetime format

    #     end_datetime_str = f"{end_date} {end_time}".replace(
    #         "-", ":"
    #     )  # Adjust to match datetime format

    #     start_time_str = start_time.replace("-", ":")
    #     end_time_str = end_time.replace("-", ":")
    #     # start_datetime = datetime.strptime(start_datetime_str, "%Y:%m:%d %H:%M:%S")
    #     # end_datetime = datetime.strptime(end_datetime_str, "%Y:%m:%d %H:%M:%S")

    #     return (
    #         start_datetime,
    #         end_datetime,
    #         start_date,
    #         start_time_str,
    #         end_date,
    #         end_time_str,
    #         duration_in_seconds,
    #     )
    # else:
    #     raise ValueError("Filename does not match the expected pattern")


#####################################################################################################################################
