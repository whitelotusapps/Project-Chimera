#####################################################################################################################################
# NATIVE MODULES
import json
from datetime import timedelta

from loguru import logger

from .date_functions import extract_date_time_from_json_filename


#####################################################################################################################################
@logger.catch
def load_json_file(file_path):
    """
    Loads and parses a JSON file from the specified file path.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        dict or list: Parsed JSON content, depending on the file structure.

    Side Effects:
        - Opens and reads a file from disk.

    Notes:
        - Assumes that the file contains valid JSON and is UTF-8 encoded.

    Caveats:
        - Raises a FileNotFoundError if the specified path does not exist.
        - Raises a json.JSONDecodeError if the file content is not valid JSON.
        - May raise an OSError for file access issues (e.g., permission denied).
    """

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def convert_to_timedelta(time_str):
    """
    Converts a string representation of seconds into a `timedelta` object.

    Args:
        time_str (str): A string representing a duration in seconds (e.g., "3600" for one hour).

    Returns:
        timedelta: A `timedelta` object representing the input duration.

    Side Effects:
        - None.

    Notes:
        - The input string is cast to a float to support fractional seconds.
        - This function is useful for converting parsed durations (e.g., from filenames or metadata) into Python-native time intervals.

    Caveats:
        - Raises a `ValueError` if the input string cannot be converted to a float.
        - Does not validate whether the duration is non-negative or within any specific range.
    """

    return timedelta(seconds=float(time_str))


#####################################################################################################################################


#####################################################################################################################################
# def format_datetime(dt):
#     """
#     Formats a `datetime` object into a string with full microsecond precision.

#     Args:
#         dt (datetime): A `datetime` object to format.

#     Returns:
#         str: A string representation of the datetime in the format
#              "YYYY-MM-DD HH-MM-SS.ffffff", where `ffffff` represents microseconds.

#     Side Effects:
#         - None.

#     Notes:
#         - Uses hyphens instead of colons for time components, which is useful for
#           file naming or systems that disallow colons in filenames (e.g., Windows).
#         - Preserves full microsecond precision, which may be useful for high-resolution
#           logging or sequencing.

#     Caveats:
#         - Assumes the input `dt` is a valid `datetime` object. Will raise an `AttributeError`
#           if an invalid object is passed.
#     """

#     return dt.strftime("%Y-%m-%d %H-%M-%S.%f")


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def calculate_duration(start, end, key_prefix):
    """
    Calculates the duration between two datetime objects and returns a dictionary
    with duration components, using a dynamic key prefix.

    Args:
        start (datetime): The starting datetime.
        end (datetime): The ending datetime.
        key_prefix (str): A prefix to prepend to each key in the resulting dictionary.

    Returns:
        dict: A dictionary containing the following duration components:
            - <key_prefix>_duration_hours (int)
            - <key_prefix>_duration_minutes (int)
            - <key_prefix>_duration_seconds (int)
            - <key_prefix>_duration_milliseconds (int)
            - <key_prefix>_total_duration_in_milliseconds (int)

    Side Effects:
        - None.

    Notes:
        - All time units are calculated from the total elapsed time between `start` and `end`.
        - Useful for logging or structured time tracking where prefixed keys are needed
          (e.g., "session_duration_hours").

    Caveats:
        - Assumes `start` and `end` are both valid `datetime` objects.
        - If `end` precedes `start`, the result will be a negative duration, which is not handled explicitly.
    """

    duration = end - start
    milliseconds = int(duration.total_seconds() * 1000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return {
        f"{key_prefix}_duration_hours": hours,
        f"{key_prefix}_duration_minutes": minutes,
        f"{key_prefix}_duration_seconds": seconds,
        f"{key_prefix}_duration_milliseconds": milliseconds,
        f"{key_prefix}_total_duration_in_milliseconds": int(
            duration.total_seconds() * 1000
        ),
    }


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def split_text_with_time(data, max_time_diff, chunk_size, file_path):
    """
    Splits transcript data into time-bound chunks based on segment durations and word-level timing.

    Args:
        data (dict): The transcript data containing segments with start/end times and word-level info.
        max_time_diff (int or float): Maximum duration in seconds allowed per chunk before creating a new one.
        chunk_size (int): Maximum number of segments allowed per chunk before creating a new one.
        file_path (str): Path to the source file containing datetime metadata in the filename.

    Returns:
        list: A list of dictionaries, each representing a time-aligned chunk with:
            - Unique chunk and segment IDs.
            - Audio and calendar timestamps.
            - Duration metadata.
            - Original segment texts and word-level metadata.

    Side Effects:
        - Relies on global functions such as `extract_date_time_from_filename`, `convert_to_timedelta`,
          `format_datetime`, and `calculate_duration` to perform time calculations and formatting.

    Notes:
        - Each chunk is created based on time and/or segment count thresholds.
        - Word-level metadata includes duration, timestamps, and token text.
        - Segment and word indices are carefully tracked across the file, chunk, and segment scopes.
        - All date and time metadata is calculated relative to the original start time extracted from the filename.

    Caveats:
        - Assumes the filename follows a strict datetime naming pattern or raises a `ValueError`.
        - Timezone handling is not explicitly addressed.
        - Assumes the presence and correctness of "segments" and "words" in the input `data`.
        - If segments are not sorted by start time, chunking behavior may be incorrect.
    """

    (
        start_date_time,
        start_datetime_str,
        end_datetime,
        end_datetime_str,
        start_date,
        unused_start_time,
        end_date,
        unused_end_time,
        duration_in_seconds,
    ) = extract_date_time_from_json_filename(file_path)

    chunks = []
    current_chunk = []
    chunk_tags = []
    chunk_keyphrases = []
    chunk_audio_start_time_location = None

    chunk_id = 1
    file_segment_id = 1
    file_word_id = 1
    chunk_segment_id = 1
    chunk_word_id = 1

    chunk_calendar_start_datetime = None  # track outside for rollover condition

    for segment in data["segments"]:
        segment_start_time_location = segment["start"]
        segment_end_time_location = segment["end"]
        segment_text = str(segment["text"]).strip()
        segment_words = segment["words"]

        segment_calendar_start_datetime = start_datetime_str + convert_to_timedelta(
            segment_start_time_location
        )
        segment_calendar_end_datetime = start_datetime_str + convert_to_timedelta(
            segment_end_time_location
        )
        segment_duration = calculate_duration(
            segment_calendar_start_datetime,
            segment_calendar_end_datetime,
            key_prefix="segment",
        )

        # Check chunk rollover BEFORE building the segment
        if not current_chunk:
            chunk_audio_start_time_location = segment_start_time_location
            chunk_calendar_start_datetime = segment_calendar_start_datetime
            chunk_segment_id = 1  # Reset for new chunk
            chunk_word_id = 1

        elif (
            segment_calendar_end_datetime - chunk_calendar_start_datetime
        ).total_seconds() > max_time_diff or len(current_chunk) >= chunk_size:
            chunk_audio_end_time_location = current_chunk[-1][
                "segment_audio_end_time_location"
            ]
            chunk_calendar_end_datetime = start_datetime_str + convert_to_timedelta(
                chunk_audio_end_time_location
            )
            chunk_duration = calculate_duration(
                chunk_calendar_start_datetime,
                chunk_calendar_end_datetime,
                key_prefix="chunk",
            )

            transcription_time_data = {
                "transcription_time_data": {
                    "chunk_audio_start_time_location": chunk_audio_start_time_location,
                    "chunk_calendar_start_datetime": chunk_calendar_start_datetime.isoformat(),
                    "chunk_audio_end_time_location": chunk_audio_end_time_location,
                    "chunk_calendar_end_datetime": chunk_calendar_end_datetime.isoformat(),
                    **chunk_duration,
                }
            }
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_tags": chunk_tags,
                    "chunk_keyphrases": chunk_keyphrases,
                    **transcription_time_data,
                    # "chunk_audio_start_time_location": chunk_audio_start_time_location,
                    # "chunk_calendar_start_datetime": format_datetime(
                    #     chunk_calendar_start_datetime
                    # ),
                    # "chunk_calendar_start_datetime": chunk_calendar_start_datetime.isoformat(),
                    # "chunk_audio_end_time_location": chunk_audio_end_time_location,
                    # "chunk_calendar_end_datetime": format_datetime(
                    #     chunk_calendar_end_datetime
                    # ),
                    # "chunk_calendar_end_datetime": chunk_calendar_end_datetime.isoformat(),
                    "chunk_text": " ".join(
                        [seg["segment_text"] for seg in current_chunk]
                    ),
                    "segments": current_chunk,
                }
            )

            # Start new chunk
            chunk_id += 1
            current_chunk = []
            chunk_audio_start_time_location = segment_start_time_location
            chunk_calendar_start_datetime = segment_calendar_start_datetime
            chunk_segment_id = 1  # Reset for new chunk
            chunk_word_id = 1

        # Now that rollover has been handled, build the segment with the correct chunk_id
        word_data = []
        segment_word_id = 1  # Reset for each new segment

        for word in segment_words:
            word_start_datetime = start_datetime_str + convert_to_timedelta(
                word["start"]
            )
            word_end_datetime = start_datetime_str + convert_to_timedelta(word["end"])
            word_duration = calculate_duration(
                word_start_datetime, word_end_datetime, key_prefix="word"
            )
            word_data.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_word_id": chunk_word_id,
                    "chunk_segment_id": chunk_segment_id,
                    "segment_word_id": segment_word_id,
                    "file_segment_id": file_segment_id,
                    "file_word_id": file_word_id,
                    "word_audio_start_time_location": word["start"],
                    # "word_calendar_start_datetime": format_datetime(
                    #     word_start_datetime
                    # ),
                    "word_calendar_start_datetime": word_start_datetime.isoformat(),
                    "word_audio_end_time_location": word["end"],
                    # "word_calendar_end_datetime": format_datetime(word_end_datetime),
                    "word_calendar_end_datetime": word_end_datetime.isoformat(),
                    **word_duration,
                    "word_text": word["word"].strip(),
                    "probability": word["probability"],
                }
            )
            file_word_id += 1
            segment_word_id += 1
            chunk_word_id += 1

        segment_data = {
            "chunk_id": chunk_id,
            "chunk_segment_id": chunk_segment_id,
            "file_segment_id": file_segment_id,
            "segment_audio_start_time_location": segment_start_time_location,
            # "segment_calendar_start_datetime": format_datetime(
            #     segment_calendar_start_datetime
            # ),
            "segment_calendar_start_datetime": segment_calendar_start_datetime.isoformat(),
            "segment_audio_end_time_location": segment_end_time_location,
            # "segment_calendar_end_datetime": format_datetime(
            #     segment_calendar_end_datetime
            # ),
            "segment_calendar_end_datetime": segment_calendar_end_datetime.isoformat(),
            **segment_duration,
            "segment_text": segment_text,
            "words": word_data,
        }

        file_segment_id += 1
        chunk_segment_id += 1
        current_chunk.append(segment_data)

    # Handle the last chunk
    if current_chunk:
        chunk_audio_end_time_location = current_chunk[-1][
            "segment_audio_end_time_location"
        ]
        chunk_calendar_end_datetime = start_datetime_str + convert_to_timedelta(
            chunk_audio_end_time_location
        )
        chunk_duration = calculate_duration(
            chunk_calendar_start_datetime,
            chunk_calendar_end_datetime,
            key_prefix="chunk",
        )

        transcription_time_data = {
            "transcription_time_data": {
                "chunk_audio_start_time_location": chunk_audio_start_time_location,
                "chunk_calendar_start_datetime": chunk_calendar_start_datetime.isoformat(),
                "chunk_audio_end_time_location": chunk_audio_end_time_location,
                "chunk_calendar_end_datetime": chunk_calendar_end_datetime.isoformat(),
                **chunk_duration,
            }
        }
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_tags": chunk_tags,
                "chunk_keyphrases": chunk_keyphrases,
                **transcription_time_data,
                # "chunk_audio_start_time_location": chunk_audio_start_time_location,
                # "chunk_calendar_start_datetime": format_datetime(
                #     chunk_calendar_start_datetime
                # ),
                # "chunk_calendar_start_datetime": chunk_calendar_start_datetime.isoformat(),
                # "chunk_audio_end_time_location": chunk_audio_end_time_location,
                # "chunk_calendar_end_datetime": format_datetime(
                #     chunk_calendar_end_datetime
                # ),
                # "chunk_calendar_end_datetime": chunk_calendar_end_datetime.isoformat(),
                "chunk_text": " ".join([seg["segment_text"] for seg in current_chunk]),
                "segments": current_chunk,
            }
        )

        # chunks.append(
        #     {
        #         "chunk_id": chunk_id,
        #         "chunk_tags": chunk_tags,
        #         "chunk_keyphrases": chunk_keyphrases,
        #         "chunk_audio_start_time_location": chunk_audio_start_time_location,
        #         # "chunk_calendar_start_datetime": format_datetime(
        #         #     chunk_calendar_start_datetime
        #         # ),
        #         "chunk_calendar_start_datetime": chunk_calendar_start_datetime.isoformat(),
        #         "chunk_audio_end_time_location": chunk_audio_end_time_location,
        #         # "chunk_calendar_end_datetime": format_datetime(
        #         #     chunk_calendar_end_datetime
        #         # ),
        #         "chunk_calendar_end_datetime": chunk_calendar_end_datetime.isoformat(),
        #         **chunk_duration,
        #         "chunk_text": " ".join([seg["segment_text"] for seg in current_chunk]),
        #         "segments": current_chunk,
        #     }
        # )

    return chunks


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def create_chunk_data(file_path, max_time_diff=120, chunk_size=512):
    """
    Creates time-based chunks from a JSON transcript file.

    Args:
        file_path (str): Path to the input JSON file containing transcript data with timing and text information.
        max_time_diff (int, optional): Maximum duration in seconds allowed per chunk. Defaults to 120.
        chunk_size (int, optional): Maximum number of segments per chunk. Defaults to 512.

    Returns:
        list: A list of chunk dictionaries, each containing metadata, text, and nested segment and word details.

    Side Effects:
        - Calls external functions `load_json_file` and `split_text_with_time` to process and segment the file data.

    Notes:
        - Uses filename-based timestamps to assign calendar-aligned timing to chunks.
        - Applies both time and segment-count constraints when deciding chunk boundaries.
        - Each chunk includes structured segment and word-level timing, text, and duration data.

    Caveats:
        - Assumes the file name follows a strict naming convention that encodes start/end timestamps.
        - Errors in filename formatting or JSON structure may cause runtime exceptions.
        - Relies on `load_json_file` returning valid and well-structured data (i.e., contains "segments").
    """

    file_data = load_json_file(file_path)
    file_chunks = split_text_with_time(file_data, max_time_diff, chunk_size, file_path)

    return file_chunks


#####################################################################################################################################
