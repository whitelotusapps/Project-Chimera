import os
import re
from datetime import datetime, timedelta

import ffmpeg
from loguru import logger
from mutagen.mp3 import MP3
from rich import box
from rich.table import Table

from .date_functions import get_audio_file_datetime_from_system


#####################################################################################################################################
@logger.catch
def convert_to_mp3(input_file, output_directory=None, bitrate="192k"):
    """
    Convert an audio file to MP3 format using ffmpeg, and rename the output file
    to include start and end timestamps along with the duration.

    Args:
        input_file (str): Path to the input audio file.
        output_directory (str, optional): Directory where the converted MP3 will be saved.
            Defaults to the input file's directory if None.
        bitrate (str, optional): Audio bitrate for the MP3 conversion (e.g., "192k").
            Defaults to "192k".

    Returns:
        str or None: The full path to the converted MP3 file with timestamped filename,
        or None if conversion fails.

    Side Effects:
        - Creates a temporary MP3 file during conversion, which is then renamed.
        - May overwrite existing files with the same target name.
        - Logs conversion progress and errors.

    Notes:
        - Requires `ffmpeg` and `ffmpeg-python` package installed and configured.
        - The output filename format is:
          "{start_datetime} - {end_datetime} - {duration_seconds} - {original_filename}.mp3"
        - Uses the file's creation or modification time as the start timestamp.
        - Duration is extracted from the converted MP3 file metadata.

    Caveats:
        - If the input file does not exist or is not a supported audio format, conversion will fail.
        - Permissions issues may arise when writing or renaming files in the output directory.
        - The function does not clean up the temporary file if an error occurs during renaming.
        - Timezone is local system time; no timezone info is embedded in the filename.

    Example:
        >>> convert_to_mp3("recording.wav", output_directory="/output")
        '/output/2025-07-02 - 14-30-15 - 2025-07-02 - 14-45-30 - 912 - recording.mp3'
    """
    file_name, file_extension = os.path.splitext(os.path.basename(input_file))
    temp_output_file = os.path.join(
        output_directory or os.path.dirname(input_file), f"{file_name}_temp.mp3"
    )

    # Run the conversion using ffmpeg
    try:
        logger.info(f"Converting {input_file} to {temp_output_file}")
        ffmpeg.input(input_file).output(
            temp_output_file,
            audio_bitrate=bitrate,
            format="mp3",
            acodec="libmp3lame",
            loglevel="quiet",
        ).run(overwrite_output=True)
    except ffmpeg.Error as e:
        logger.error(f"Error converting {input_file}: {e}")
        return None

    # Get file creation time and duration
    start_datetime = datetime.fromtimestamp(
        get_audio_file_datetime_from_system(input_file)
    )
    audio = MP3(temp_output_file)
    duration = int(audio.info.length)
    end_datetime = start_datetime + timedelta(seconds=duration)

    # Construct the final filename
    new_filename = f"{start_datetime.strftime('%Y-%m-%d - %H-%M-%S')} - {end_datetime.strftime('%Y-%m-%d - %H-%M-%S')} - {duration} - {file_name}.mp3"
    new_output_file = os.path.join(
        output_directory or os.path.dirname(input_file), new_filename
    )

    os.rename(temp_output_file, new_output_file)
    logger.info(f"Converted {input_file} to {new_output_file}")

    return new_output_file


#####################################################################################################################################


#####################################################################################################################################
# Define a function to create and print tables with non-wrapped titles
#####################################################################################################################################
@logger.catch
def print_table(config_path, config, console, file_console, logfile_path_with_name):
    """
    Display configuration data in a formatted table on the console and write it to a logfile.

    Args:
        config_path (str): The path or name to display as the table title.
        config (dict or other): The configuration data to display, expected to be a dictionary.

    Side Effects:
        - Prints a styled table to the console using the `rich` library.
        - Writes the same table output to a logfile defined by `logfile_path_with_name`.
        - Uses global variables `console` and `file_console` which must be properly initialized.

    Notes:
        - If `config` is not a dictionary, prints an error message in red to the console.
        - The "search_and_replace" key's value is summarized by displaying the count of pairs,
          rather than listing all items.
        - Lists in the config are displayed by showing the first item next to the key and
          subsequent items in separate rows with an empty key column for readability.
        - Adds blank rows between entries for better visual separation.

    Caveats:
        - Assumes `console`, `file_console`, and `logfile_path_with_name` are defined in the global scope.
        - Overwrites the logfile at `logfile_path_with_name` on each call.
        - If the logfile path is invalid or not writable, this will raise an error.
        - Large lists or deeply nested data may produce cluttered output.
        - This function does not return any value.

    Example:
        >>> print_table("/path/to/config.yaml", {"search_and_replace": [("foo", "bar"), ("baz", "qux")], "option": "value"})
        # Prints a formatted table to console and writes to the logfile.
    """
    table = Table(
        title=config_path, title_justify="left", expand=True, box=box.SIMPLE_HEAD
    )
    table.add_column("Key", style="cyan", no_wrap=True, width=10)
    table.add_column("Value", style="green", width=30)

    if isinstance(config, dict):
        for key, value in config.items():
            # We're not going to display all of the search in replace pairs
            # Instead, we're going to notify the user how many search and replace
            # pairs were loaded
            if key == "search_and_replace_pairs":
                value = f"{len(value)} search and replace pairs loaded"

            if key == "contractions_dict":
                value = f"{len(key)} contractions loads"

            if key == "word_cloud_variables":
                continue
            if isinstance(value, list):
                # Add the first row with the key and the first item in the list
                table.add_row(key, str(value[0]))
                # Add the remaining items in the list with an empty key column
                for item in value[1:]:
                    table.add_row("", str(item))
            else:
                # Add the row for non-list items
                table.add_row(key, str(value))
            # Add a blank row for better readability
            table.add_row("", "")
    else:
        console.print("[red]Error: Unsupported results format.[/red]")

    # Print the config values to our console
    console.print(table)

    # Write config values to our logfile
    with open(logfile_path_with_name, "w", encoding="utf-8") as file:
        file_console.file = file
        file_console.print(table)


#####################################################################################################################################


#####################################################################################################################################
# Function to perform the replacements
#####################################################################################################################################
@logger.catch
def replace_text(file_data, search_and_replace_pairs):
    """
    Perform multiple search-and-replace operations on a given text string.

    Args:
        file_data (str): The input text to perform replacements on.
        search_and_replace_pairs (list of dict): A list of dictionaries where each dictionary
            must contain the keys 'search' (a regex pattern or string to search for)
            and 'replace' (the replacement string).

    Returns:
        str: The text after all search-and-replace operations have been applied.
             Returns the original text unchanged if no pairs are provided.

    Side Effects:
        None directly on external state; purely functional on the input string.

    Notes:
        - Uses `re.sub()` for replacements, so 'search' can be a regex pattern.
        - The replacements are applied sequentially in the order of the list.
        - If `search_and_replace_pairs` is empty or None, logs a warning and returns
          the original input unchanged.
        - The function assumes `search_and_replace_pairs` is well-formed; no explicit
          validation of the dictionary keys is performed.

    Caveats:
        - If any 'search' pattern is invalid regex, `re.sub` will raise an exception.
        - Overlapping or conflicting replacements may cause unexpected results.
        - Consider escaping special regex characters in 'search' if literal matching
          is desired.
    """
    if len(search_and_replace_pairs) > 0:

        search_and_replace_file_data = file_data

        for pair in search_and_replace_pairs:
            search = pair.search
            replace = pair.replace
            # Perform the search and replace
            search_and_replace_file_data = re.sub(
                search, replace, search_and_replace_file_data
            )

        return search_and_replace_file_data
    else:
        logger.opt(colors=True).info(
            "<RED><white><b>NO SEARCH AND REPLACE PAIRS DEFINED</b></white></RED>"
        )
        return file_data


#####################################################################################################################################
# Used for rebuilding the "chunk" dictionary the way we want it, by adding new meta-data in the order we'd like to have it
#####################################################################################################################################
@logger.catch
def insert_keys(original_dict, insert_after_key, new_items):
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key] = value
        if key == insert_after_key:
            new_dict.update(new_items)
    return new_dict
