import os
from datetime import datetime
from pathlib import Path

from loguru import logger
from mutagen.flac import FLAC, Picture
from mutagen.id3 import (
    APIC,
    COMM,
    ID3,
    TALB,
    TCON,
    TDRC,
    TDTG,
    TIT1,
    TIT2,
    TIT3,
    TOFN,
    TPE1,
    TPE2,
    TRCK,
    USLT,
    ID3NoHeaderError,
)
from mutagen.mp3 import MP3

from .date_functions import get_start_date_and_end_date
from .project_paths import PATHS
from .transcript_word_cloud_functions import generate_stylecloud


#####################################################################################################################################
@logger.catch
def embed_cover_art(file_path, file_extension, image_file):
    """
    Embed cover art image into an MP3 file's ID3 tags and delete the image file afterward.

    Args:
        mp3_file (str): Path to the target MP3 audio file.
        image_file (str): Path to the image file (expected PNG) to embed as cover art.

    Side Effects:
        - Modifies the metadata (ID3 tags) of the specified MP3 file by adding or updating its cover art.
        - Deletes the specified image file from the filesystem after embedding.
        - Logs an informational message indicating successful embedding.

    Notes:
        - The function expects the image file to be in PNG format; the MIME type is hardcoded as "image/png".
        - If the MP3 file does not already contain ID3 tags, they will be added automatically.
        - The ID3 `APIC` frame type is set to 3, which corresponds to the front cover image.
        - The image file is read in binary mode and fully embedded into the MP3 file's tags.
        - The function assumes that the `mutagen` library (specifically `MP3`, `ID3`, `APIC`) is installed and imported.
        - The function will raise an exception if the MP3 or image file paths are invalid or inaccessible.
        - Deleting the image file (`os.unlink`) means the image is removed after embedding; ensure this behavior is intended.

    Caveats:
        - Only PNG images are supported as cover art in this implementation; to support other formats, MIME type and possibly other fields need adjustment.
        - Overwrites existing cover art without backing up.
        - If the MP3 file is write-protected or open elsewhere, saving tags may fail.

    Example:
        >>> embed_cover_art("song.mp3", "cover.png")
    """
    # logger.info(f"file_extension: {file_extension}")
    # input("\n\nHERE\n\n")

    if file_extension.strip().lower() == ".mp3":
        audio = MP3(file_path, ID3=ID3)

        # Add ID3 tag if not present
        if not audio.tags:
            audio.add_tags()

        with open(image_file, "rb") as img:
            audio.tags.add(
                APIC(
                    encoding=3,  # UTF-8
                    mime="image/png",  # MIME type of the image
                    type=3,  # Front cover
                    desc="Cover",
                    data=img.read(),
                )
            )

        audio.save()
        logger.info(f"Cover art embedded into {file_path}")
        # os.unlink(image_file)
        return

    if file_extension.strip().lower() == ".flac":
        audio = FLAC(file_path)

        # Create the Picture object
        picture = Picture()
        picture.type = 3  # Front cover
        picture.desc = "Cover"
        picture.mime = "image/png"

        with open(image_file, "rb") as img:
            picture.data = img.read()

        # Add picture to FLAC file
        audio.clear_pictures()  # Remove any existing cover art
        audio.add_picture(picture)
        audio.save()
        logger.info(f"Cover art embedded into {file_path}")
        # os.unlink(image_file)
        return


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def populate_mp3_tags(
    file_path,
    files_to_transcribe,
    txt_path,
    text_path,
    artist_name,
    top_n_words,
    custom_stop_words,
    corpus_extensions_pattern,
):
    """
    Parse an MP3 filename for metadata, update its ID3 tags with detailed info, generate a word cloud image
    from the transcription text, and embed the word cloud as cover art.

    Args:
        mp3_path (str): Path to the MP3 audio file to tag.
        txt_path (str): Path to the text transcription file corresponding to the MP3.
        artist_name (str): Name of the artist to embed in the tags.
        files_to_transcribe (list): List of all MP3 files (paths) for the same recording date.
        script_path (str): Base script directory path used to locate assets and temp folders.
        TOP_N_WORDS (int): Number of top words to include in the generated style cloud.
        CUSTOM_STOPWORDS (set or None): Optional set of custom stopwords to exclude from the word cloud.

    Side Effects:
        - Modifies the ID3 tags of the specified MP3 file, setting fields including title, artist, album,
          track number, year, genre, comments, and lyrics (transcription).
        - Generates a style cloud PNG image from the transcription text and saves it in a temp folder.
        - Embeds the generated word cloud image as cover art in the MP3 file.
        - Logs multiple informational messages throughout processing.
        - Skips files not matching the expected filename pattern or missing transcription text files.

    Notes:
        - The MP3 filename must match the pattern:
          "YYYY-MM-DD - HH-MM-SS - YYYY-MM-DD - HH-MM-SS - <number> - <comment>.mp3".
        - Track index and total track count are determined by comparing MP3 files sharing the same recording date.
        - Date and time are extracted and reformatted to update recording date tag (TDRC).
        - If existing ID3 tags are present, they are deleted before adding new ones.
        - The transcription text is loaded and embedded as unsynchronized lyrics (USLT) in English.
        - The generated style cloud uses TF-IDF scoring and custom stopwords to highlight prominent words.
        - The function assumes required libraries (`mutagen`, `logging`, `re`, `datetime`, etc.) are imported.
        - The function expects the `generate_stylecloud` and `embed_cover_art` utilities to be defined elsewhere.

    Caveats:
        - Will silently skip files whose names do not match the regex pattern.
        - Will skip processing if the transcription text file does not exist.
        - Assumes transcription text encoding is UTF-8.
        - The temporary word cloud image file is stored inside the `assets/temp` folder within `script_path`.
        - Overwrites any existing tags on the MP3 file without backup.
        - Assumes MP3 and text files are accessible and writable.
        - Logging verbosity depends on the logger configuration.

    Example:
        >>> tag_mp3_with_metadata(
                mp3_path="2023-06-15 - 14-30-00 - 2023-06-15 - 14-35-00 - 1 - morning_journal.mp3",
                txt_path="2023-06-15_morning_journal.txt",
                artist_name="John Doe",
                files_to_transcribe=[...],
                script_path="/home/user/project",
                TOP_N_WORDS=100,
                CUSTOM_STOPWORDS={"um", "uh", "like"}
            )
    """
    # pattern = re.compile(
    #     r"^(\d{4}-\d{2}-\d{2}) - (\d{2})-\d{2}-\d{2} - \d{4}-\d{2}-\d{2} - \d{2}-\d{2}-\d{2} - \d+ - .+\.mp3$"
    # )
    # mp3_filename = os.path.basename(mp3_path)

    # match = pattern.match(mp3_filename)
    # if not match:
    #     logger.info(f"Skipping file with unexpected format: {mp3_filename}")
    #     return

    # logger.info(f"file_path: {file_path}")
    # logger.info(f"file_name: {file_path.name}")
    start_date, start_time, end_date, end_time, comment, extension = (
        get_start_date_and_end_date(file_path.name, corpus_extensions_pattern)
    )

    # logger.info(f"start_date: {start_date}")
    # logger.info(f"start_time: {start_time}")
    # logger.info(f"end_date: {end_date}")
    # logger.info(f"end_time: {end_time}")
    # logger.info(f"comment: {comment}")
    # logger.info(f"extension: {extension}")

    # input("\n\nHERE\n\n")

    # Extract album_title as the first part before the second timestamp
    track_title = f"{start_date} - {start_time} - {end_date} - {end_time}"

    # track_title = match.group(0)[:-4]  # Keep track_title unchanged
    album_year, album_month, album_day = start_date.split("-")
    album_title = start_date

    # Determine track placement and total tracks for the recording date
    files_for_date = [
        f.name for f in files_to_transcribe if Path(f).name.startswith(album_title)
    ]

    files_for_date.sort()

    logger.info(
        "files_to_transcribe:\n\n\t"
        + "\n\t".join(str(p) for p in files_to_transcribe)
        + "\n"
    )
    logger.info(
        "files_for_date (sorted):\n\n\t"
        + "\n\t".join(str(p) for p in files_for_date)
        + "\n"
    )

    total_tracks = len(files_for_date)

    # track_index = files_for_date.index(mp3_filename) + 1 if mp3_filename in files_for_date else 1
    track_index = (
        files_for_date.index(file_path.name) + 1
        if file_path.name in files_for_date
        else 1
    )

    # logger.info(f"file_path.name: {file_path.name}")
    # logger.info(f"files_for_date: {files_for_date}")
    # logger.info(f"track_index: {track_index}")
    # logger.info(f"total_tracks: {total_tracks}")

    # input("\nHERE\n")
    # Extract the first date-time segment: "YYYY-MM-DD - HH-MM-SS"
    recording_time_raw = (
        " ".join(track_title.split(" - ")[:2]).replace("-", ":", 2).replace("-", " ", 1)
    )

    if not txt_path.exists():
        logger.info(f"Missing text file: {txt_path}, skipping {file_path.name}")
        return

    with open(txt_path, "r", encoding="utf-8") as f:
        data = f.read()

    try:
        audio = MP3(file_path, ID3=ID3)
        if audio.tags:
            del audio.tags  # Remove all existing ID3 tags
        audio.tags = ID3()
    except ID3NoHeaderError:
        audio = MP3(file_path)
        audio.tags = ID3()

    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")

    # Update ID3 tags
    audio.tags["TIT2"] = TIT2(encoding=3, text=track_title)
    audio.tags["TPE1"] = TPE1(encoding=3, text=artist_name)
    audio.tags["TRCK"] = TRCK(
        encoding=3, text=f"{track_index}/{total_tracks}"
    )  # Track number / Total tracks
    audio.tags["TALB"] = TALB(encoding=3, text=album_title)
    audio.tags["TDRC"] = TDRC(encoding=3, text=recording_time_raw)
    audio.tags["TCON"] = TCON(encoding=3, text="Audio Journal")
    audio.tags["TPE2"] = TPE2(encoding=3, text=artist_name)
    audio.tags["COMM"] = COMM(encoding=3, lang="eng", desc="", text=comment)
    audio.tags["TOFN"] = TOFN(encoding=3, text=os.path.basename(txt_path))
    audio.tags["TIT1"] = TIT1(encoding=3, text=album_year)
    audio.tags["TIT3"] = TIT3(encoding=3, text=album_month)
    audio.tags["TDTG"] = TDTG(encoding=3, text=date_string)
    audio.tags["USLT"] = USLT(encoding=3, lang="eng", desc="Transcription", text=data)

    audio.save()
    logger.info(
        f"Updated tags for {file_path.name} (Track {track_index}/{total_tracks})"
    )

    output_image = PATHS.temp / f"{file_path.stem}_wordcloud.png"

    logger.info(f"output_image: {output_image}")

    generate_stylecloud(text_path, output_image, top_n_words, custom_stop_words)

    if output_image.exists():
        embed_cover_art(file_path, extension, output_image)


#####################################################################################################################################


@logger.catch
def populate_flac_tags(
    file_path,
    files_to_transcribe,
    txt_path,
    text_path,
    artist_name,
    top_n_words,
    custom_stop_words,
    corpus_extensions_pattern,
):
    """
    Parse an MP3 filename for metadata, update its ID3 tags with detailed info, generate a word cloud image
    from the transcription text, and embed the word cloud as cover art.

    Args:
        mp3_path (str): Path to the MP3 audio file to tag.
        txt_path (str): Path to the text transcription file corresponding to the MP3.
        artist_name (str): Name of the artist to embed in the tags.
        files_to_transcribe (list): List of all MP3 files (paths) for the same recording date.
        script_path (str): Base script directory path used to locate assets and temp folders.
        TOP_N_WORDS (int): Number of top words to include in the generated style cloud.
        CUSTOM_STOPWORDS (set or None): Optional set of custom stopwords to exclude from the word cloud.

    Side Effects:
        - Modifies the ID3 tags of the specified MP3 file, setting fields including title, artist, album,
          track number, year, genre, comments, and lyrics (transcription).
        - Generates a style cloud PNG image from the transcription text and saves it in a temp folder.
        - Embeds the generated word cloud image as cover art in the MP3 file.
        - Logs multiple informational messages throughout processing.
        - Skips files not matching the expected filename pattern or missing transcription text files.

    Notes:
        - The MP3 filename must match the pattern:
          "YYYY-MM-DD - HH-MM-SS - YYYY-MM-DD - HH-MM-SS - <number> - <comment>.mp3".
        - Track index and total track count are determined by comparing MP3 files sharing the same recording date.
        - Date and time are extracted and reformatted to update recording date tag (TDRC).
        - If existing ID3 tags are present, they are deleted before adding new ones.
        - The transcription text is loaded and embedded as unsynchronized lyrics (USLT) in English.
        - The generated style cloud uses TF-IDF scoring and custom stopwords to highlight prominent words.
        - The function assumes required libraries (`mutagen`, `logging`, `re`, `datetime`, etc.) are imported.
        - The function expects the `generate_stylecloud` and `embed_cover_art` utilities to be defined elsewhere.

    Caveats:
        - Will silently skip files whose names do not match the regex pattern.
        - Will skip processing if the transcription text file does not exist.
        - Assumes transcription text encoding is UTF-8.
        - The temporary word cloud image file is stored inside the `assets/temp` folder within `script_path`.
        - Overwrites any existing tags on the MP3 file without backup.
        - Assumes MP3 and text files are accessible and writable.
        - Logging verbosity depends on the logger configuration.

    Example:
        >>> tag_mp3_with_metadata(
                mp3_path="2023-06-15 - 14-30-00 - 2023-06-15 - 14-35-00 - 1 - morning_journal.mp3",
                txt_path="2023-06-15_morning_journal.txt",
                artist_name="John Doe",
                files_to_transcribe=[...],
                script_path="/home/user/project",
                TOP_N_WORDS=100,
                CUSTOM_STOPWORDS={"um", "uh", "like"}
            )
    """
    # pattern = re.compile(
    #     r"^(\d{4}-\d{2}-\d{2}) - (\d{2})-\d{2}-\d{2} - \d{4}-\d{2}-\d{2} - \d{2}-\d{2}-\d{2} - \d+ - .+\.mp3$"
    # )
    # mp3_filename = os.path.basename(mp3_path)

    # match = pattern.match(mp3_filename)
    # if not match:
    #     logger.info(f"Skipping file with unexpected format: {mp3_filename}")
    #     return

    # logger.info(f"file_path: {file_path}")
    # logger.info(f"file_name: {file_path.name}")
    start_date, start_time, end_date, end_time, comment, extension = (
        get_start_date_and_end_date(file_path.name, corpus_extensions_pattern)
    )

    # logger.info(f"start_date: {start_date}")
    # logger.info(f"start_time: {start_time}")
    # logger.info(f"end_date: {end_date}")
    # logger.info(f"end_time: {end_time}")
    # logger.info(f"comment: {comment}")
    # logger.info(f"extension: {extension}")

    # input("\n\nHERE\n\n")

    # Extract album_title as the first part before the second timestamp
    track_title = f"{start_date} - {start_time} - {end_date} - {end_time}"

    # track_title = match.group(0)[:-4]  # Keep track_title unchanged
    album_year, album_month, album_day = start_date.split("-")
    album_title = start_date

    # Determine track placement and total tracks for the recording date
    files_for_date = [
        f.name for f in files_to_transcribe if Path(f).name.startswith(album_title)
    ]

    files_for_date.sort()

    logger.info(
        "files_to_transcribe:\n\n\t"
        + "\n\t".join(str(p) for p in files_to_transcribe)
        + "\n"
    )
    logger.info(
        "files_for_date (sorted):\n\n\t"
        + "\n\t".join(str(p) for p in files_for_date)
        + "\n"
    )

    total_tracks = len(files_for_date)

    # track_index = files_for_date.index(mp3_filename) + 1 if mp3_filename in files_for_date else 1
    track_index = (
        files_for_date.index(file_path.name) + 1
        if file_path.name in files_for_date
        else 1
    )

    logger.info(f"track_index: {track_index}")
    logger.info(f"total_tracks: {total_tracks}")
    # input("\nHERE\n")
    # Extract the first date-time segment: "YYYY-MM-DD - HH-MM-SS"
    recording_time_raw = (
        " ".join(track_title.split(" - ")[:2]).replace("-", ":", 2).replace("-", " ", 1)
    )

    if not txt_path.exists():
        logger.info(f"Missing text file: {txt_path}, skipping {file_path.name}")
        return

    with open(txt_path, "r", encoding="utf-8") as f:
        data = f.read()

    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")

    audio = FLAC(file_path)
    # Clear existing tags
    audio.delete()

    # Add Vorbis comments
    audio["TITLE"] = track_title
    audio["ARTIST"] = artist_name
    audio["TRACKNUMBER"] = str(track_index)
    audio["TRACKTOTAL"] = str(total_tracks)
    audio["ALBUM"] = album_title
    audio["DATE"] = recording_time_raw
    audio["GENRE"] = "Audio Journal"
    audio["ALBUMARTIST"] = artist_name
    audio["COMMENT"] = comment
    audio["ORIGINALFILENAME"] = txt_path.name
    audio["ALBUM_YEAR"] = album_year
    audio["ALBUM_MONTH"] = album_month
    audio["TAGGING_DATE"] = date_string
    audio["TRANSCRIPTION"] = data

    audio.save()

    logger.info(
        f"Updated tags for {file_path.name} (Track {track_index}/{total_tracks})"
    )
    output_image = PATHS.temp / f"{file_path.stem}_wordcloud.png"

    logger.info(f"output_image: {output_image}")

    generate_stylecloud(text_path, output_image, top_n_words, custom_stop_words)

    if output_image.exists():
        embed_cover_art(file_path, extension, output_image)


#####################################################################################################################################
