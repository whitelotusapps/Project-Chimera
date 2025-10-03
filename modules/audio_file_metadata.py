# THIS FILE GENERATES THE PER FILE JSON FILE OF DATA OBTAINED FROM THE pymediainfo MODULE
# THIS FILE ALSO GENERATES THE MASTER CSV FILE FOR THE CORPUS

import hashlib
import re
from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from loguru import logger
from pymediainfo import MediaInfo

#####################################################################################################################################
@logger.catch
def flatten_json(json_data, ignore_keys=None):
    ignore_keys = set(ignore_keys or [])
    flat_data = OrderedDict()  # Ordered to preserve key order

    for key, value in json_data.items():
        if key in ignore_keys:
            continue

        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_data[f"{key}.{subkey}"] = subvalue
        else:
            flat_data[key] = value

    return flat_data
#####################################################################################################################################

#####################################################################################################################################
@logger.catch
def format_full_datetime(utc_datetime_str, timezone_label):
    # Remove the ' UTC' suffix and parse into datetime
    dt = datetime.strptime(utc_datetime_str.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")

    # Extract the day and get the correct suffix
    day = dt.day
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    # Format with zero-padded day and hour (cross-platform safe)
    formatted = dt.strftime("%A, %B {day}, %Y %I:%M%p").format(day=day)

    # Remove leading zero from the hour (e.g., '05:18PM' -> '5:18PM')
    formatted = re.sub(r"(\D)0(\d:)", r"\1\2", formatted)

    # Add the day suffix
    formatted = formatted.replace(f"{day},", f"{day}{suffix},")

    # Add the time zone label
    # full_formatted = f"{formatted} {timezone_label}"

    # return full_formatted

    return formatted
#####################################################################################################################################

#####################################################################################################################################
@logger.catch
def get_utc_offset_and_us_timezone(local_str, utc_str):
    # Parse the datetime strings to naive datetime objects
    local_dt = datetime.strptime(local_str, "%Y-%m-%d %H:%M:%S.%f")
    utc_dt = datetime.strptime(utc_str.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S.%f")

    # Calculate the UTC offset as a timedelta
    offset = local_dt - utc_dt  # timedelta

    # Format offset as ±H:MM string
    total_minutes = int(offset.total_seconds() / 60)
    sign = "-" if total_minutes < 0 else "+"
    abs_minutes = abs(total_minutes)
    hours = abs_minutes // 60
    minutes = abs_minutes % 60
    offset_str = f"{sign}{hours}:{minutes:02d}"

    # US time zones to check (from zoneinfo available zones, filtered)
    us_zones = [
        "America/New_York",  # Eastern Time (UTC-5 or UTC-4 DST)
        "America/Chicago",  # Central Time (UTC-6 or UTC-5 DST)
        "America/Denver",  # Mountain Time (UTC-7 or UTC-6 DST)
        "America/Phoenix",  # Mountain Standard Time (no DST, UTC-7)
        "America/Los_Angeles",  # Pacific Time (UTC-8 or UTC-7 DST)
        "America/Anchorage",  # Alaska Time (UTC-9 or UTC-8 DST)
        "Pacific/Honolulu",  # Hawaii-Aleutian Time (UTC-10 no DST)
    ]

    # Find which US time zone matches the offset at the local datetime
    matched_zone = None
    for zone_name in us_zones:
        tz = ZoneInfo(zone_name)
        # Attach tzinfo to local_dt without changing the clock time (assume local_dt is in that timezone)
        local_dt_tz = local_dt.replace(tzinfo=tz)
        # Calculate offset from tzinfo
        tz_offset = local_dt_tz.utcoffset()
        if tz_offset == offset:
            matched_zone = zone_name
            break

    # If no exact match, fallback to closest zone by difference in offset (optional)
    if matched_zone is None:

        def offset_diff(zone_name):
            tz = ZoneInfo(zone_name)
            local_dt_tz = local_dt.replace(tzinfo=tz)
            tz_offset = local_dt_tz.utcoffset()
            return abs((tz_offset - offset).total_seconds())

        matched_zone = min(us_zones, key=offset_diff)

    # Friendly names for US zones
    friendly_names = {
        "America/New_York": "Eastern Time",
        "America/Chicago": "Central Time",
        "America/Denver": "Mountain Time",
        "America/Phoenix": "Mountain Standard Time",
        "America/Los_Angeles": "Pacific Time",
        "America/Anchorage": "Alaska Time",
        "Pacific/Honolulu": "Hawaii-Aleutian Time",
    }
    friendly_name = friendly_names.get(matched_zone, "Unknown Time")

    return offset_str, matched_zone, friendly_name
#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def compute_sha256(file_path):
    logger.info("Calculating SHA256...")
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()
#####################################################################################################################################

#####################################################################################################################################
@logger.catch
def extract_mp3_info(file_path, sha256_hash):
    file_general_file_name_extension = ""
    file_general_complete_name = ""
    file_general_folder_name = ""
    file_general_number_of_audio_streams = ""
    file_general_number_of_image_streams = ""
    file_general_audio_codec = ""
    file_general_image_codec = ""
    file_general_internet_media_type = ""
    file_general_total_file_size_in_bytes = ""
    other_file_size_list = ""
    file_general_total_file_size_pretty = ""
    file_general_total_duration_in_milliseconds = ""
    general_other_duration_list = ""
    file_general_total_duration_timestamp = ""
    file_general_overall_bitrate = ""
    other_overall_bit_rate_list = ""
    file_general_overall_bitrate_pretty = ""
    file_general_stream_size_in_bytes = ""
    other_stream_size_list = ""
    file_general_stream_size_pretty = ""
    file_general_proportion_of_this_stream = ""

    ###########################################
    # FILE DATE SPECIFIC DATA
    file_general_recorded_date_utc = ""
    file_general_tagged_date_utc = ""
    file_general_file_creation_date_utc = ""
    file_general_file_creation_date__local = ""
    file_gerneral_file_last_modification_date_utc = ""
    file_general_file_last_modification_date__local = ""

    ###########################################
    # TRACK SPECIFIC DATA
    file_general_track_title = ""
    file_general_track_album = ""
    file_general_track_album_performer = ""
    file_general_track_name = ""
    file_general_track_name_position = ""
    file_general_track_name_total = ""
    file_general_track_more = ""
    file_general_track_grouping = ""
    file_general_track_performer = ""
    file_general_track_genre = ""

    ###########################################
    # MISC GENERAL DATA
    file_general_track_writing_library = ""
    file_general_comment = ""
    file_general_id3v1_comment = ""

    ###########################################
    #  TRANSCRIPT GENERAL DATA
    file_general_lyrics = ""
    file_general_original_filename = ""

    ###########################################
    # GENERAL COVER ART DATA
    file_general_has_cover = ""
    file_general_cover_description = ""
    file_general_cover_type = ""
    file_general_cover_mime = ""

    ###########################################
    # AUDIO STREAM SPECIFIC DATA
    file_audio_commercial_name = ""
    file_audio_format_version = ""
    file_audio_format_profile = ""
    file_audio_total_duration_in_milliseconds = ""
    audio_other_duration_list = ""
    file_audio_total_duration_timestamp = ""
    file_audio_bit_rate_mode = ""
    audio_other_bit_rate_mode_list = ""
    file_audio_bit_rate_mode_pretty = ""
    file_audio_bit_rate = ""
    audio_other_bit_rate_list = ""
    file_audio_bit_rate_pretty = ""
    file_audio_channel_s = ""
    audio_other_channel_s_list = ""
    file_audio_channel_s_pretty = ""
    file_audio_samples_per_frame = ""
    file_audio_sampling_rate = ""
    audio_other_sampling_rate_list = ""
    file_audio_sampling_rate_pretty = ""
    file_audio_samples_count = ""
    file_audio_frame_rate = ""
    audio_other_frame_rate_list = ""
    file_audio_frame_rate_pretty = ""
    file_audio_frame_count = ""
    file_audio_compression_mode = ""
    file_audio_stream_size_in_bytes = ""
    audio_other_stream_size_list = ""
    file_audio_stream_size_pretty = ""
    file_audio_proportion_of_this_stream = ""

    ###########################################
    # IMAGE STREAM SPECIFIC DATA
    file_image_format_info = ""
    file_image_commercial_name = ""
    file_image_compression = ""
    file_image_format_settings = ""
    file_image_internet_media_type = ""
    file_image_width = ""
    file_image_height = ""
    file_image_pixel_aspect_ratio = ""
    file_image_display_aspect_ratio = ""
    file_image_color_space = ""
    file_image_bit_depth = ""
    image_other_bit_depth_list = ""
    file_image_bit_depth_pretty = ""
    file_image_compression_mode = ""
    file_image_stream_size_in_bytes = ""
    image_other_stream_size = ""
    file_image_stream_size_pretty = ""
    file_image_proportion_of_this_stream = ""
    full_recording_date_and_time = ""

    utc_offset = ""
    matched_tz = ""
    tz_friendly_name = ""

    full_recording_date_and_time = ""

    logger.info("Extracting media info...")
    media_info_obj = MediaInfo.parse(file_path)
    media_info = media_info_obj.to_data()

    tracks = media_info.get("tracks", [])

    for track in tracks:
        if track.get("track_type") == "General":

            ###########################################
            # TECHNICAL FILE SPECIFIC DATA

            file_general_file_name_extension = track.get(
                "file_name_extension", None
            )  # 2025-07-04 - 17-18-37 - 2025-07-04 - 17-19-06 - 29 - audio journal - TEST.mp3

            file_general_complete_name = track.get(
                "complete_name", None
            )  # C:\\temp\\audio_temp\\mp3_metadata_test\\2025-07-04 - 17-18-37 - 2025-07-04 - 17-19-06 - 29 - audio journal - TEST.mp3

            file_general_folder_name = track.get(
                "folder_name", None
            )  # C:\\temp\\audio_temp\\mp3_metadata_test

            file_general_number_of_audio_streams = track.get(
                "count_of_audio_streams", None
            )  # 1
            file_general_number_of_image_streams = track.get(
                "count_of_image_streams", None
            )  # 1
            file_general_audio_codec = track.get("audio_codecs", None)  # MPEG Audio
            file_general_image_codec = track.get("codecs_image", None)  # PNG
            file_general_internet_media_type = track.get(
                "internet_media_type", None
            )  # audio/mpeg

            file_general_total_file_size_in_bytes = track.get(
                "file_size", None
            )  # 702945
            other_file_size_list = track.get("other_file_size", [])
            if len(other_file_size_list) > 0:
                file_general_total_file_size_pretty = other_file_size_list[4]  # 686.5 KiB

            file_general_total_duration_in_milliseconds = track.get("duration")  # 29387
            general_other_duration_list = track.get("other_duration", [])
            if len(general_other_duration_list) > 0:
                file_general_total_duration_timestamp = general_other_duration_list[
                    4
                ]  # 00:00:29.387

            file_general_overall_bitrate = track.get("overall_bit_rate", None)  # 128000
            other_overall_bit_rate_list = track.get("other_overall_bit_rate", [])
            if len(other_overall_bit_rate_list) > 0:
                file_general_overall_bitrate_pretty = other_overall_bit_rate_list[
                    0
                ]  # 128 kb/s

            file_general_stream_size_in_bytes = track.get("stream_size", None)
            other_stream_size_list = track.get("other_stream_size", [])
            if len(other_stream_size_list) > 0:
                file_general_stream_size_pretty = other_stream_size_list[0]  # 227 KiB (33%)

            file_general_proportion_of_this_stream = track.get(
                "proportion_of_this_stream", None
            )  # Raw number that could be converted to percentage (0.33110)

            ###########################################
            # FILE DATE SPECIFIC DATA
            file_general_recorded_date_utc = track.get(
                "recorded_date", None
            )  # 2025-07-04 17:18:37 UTC
            file_general_tagged_date_utc = track.get(
                "tagged_date", None
            )  # 2025-07-05 15:21:07 UTC
            file_general_file_creation_date_utc = track.get(
                "file_creation_date", None
            )  # 2025-07-04 22:22:50.460 UTC
            file_general_file_creation_date__local = track.get(
                "file_creation_date__local", None
            )  # 2025-07-04 17:22:50.460
            file_gerneral_file_last_modification_date_utc = track.get(
                "file_last_modification_date", None
            )  # 2025-07-05 20:21:10.740 UTC
            file_general_file_last_modification_date__local = track.get(
                "file_last_modification_date__local", None
            )  # 2025-07-05 15:21:10.740

            ###########################################
            # TRACK SPECIFIC DATA
            file_general_track_title = track.get(
                "title", None
            )  # 2025-07-04 - 17-18-37 - 2025-07-04 - 17-19-06
            file_general_track_album = track.get(
                "album", None
            )  # 2025-07-04 / What is the date this audio journal is related to?
            file_general_track_album_performer = track.get(
                "album_performer", None
            )  # The Real Zack Olinger / Who performed this album?
            file_general_track_name = track.get(
                "track_name", None
            )  # 2025-07-04 - 17-18-37 - 2025-07-04 - 17-19-06
            file_general_track_name_position = track.get(
                "track_name_position", None
            )  # 2 / Which track is this on this album?
            file_general_track_name_total = track.get(
                "track_name_total", None
            )  # 2 / How many tracks total on this album?

            file_general_track_more = track.get(
                "track_more", None
            )  # 07 / What month of the year is this track from?

            file_general_track_grouping = track.get(
                "grouping", None
            )  # 2025  / What year is this track from?
            file_general_track_performer = track.get(
                "performer", None
            )  # The Real Zack Olinger / Who performed this track?

            file_general_track_genre = track.get("genre", None)  # Audio Journal

            ###########################################
            # MISC GENERAL DATA
            file_general_track_writing_library = track.get(
                "writing_library", None
            )  # LAME3.10

            file_general_comment = track.get(
                "comment", None
            )  # 29 - audio journal - TEST / What is left of the original file name?
            file_general_id3v1_comment = track.get(
                "id3v1_comment", None
            )  # 29 - audio journal - TEST / What is left of the original file name?

            ###########################################
            #  TRANSCRIPT GENERAL DATA
            file_general_lyrics = track.get("lyrics", None)  # Embeded transcript text
            file_general_original_filename = track.get(
                "original_filename", None
            )  # 2025-07-04 - 17-18-37 - 2025-07-04 - 17-19-06 - 29 - audio journal - TEST - large-v2 - SR.txt / Where did the text for the transcript comee from?

            ###########################################
            # GENERAL COVER ART DATA
            file_general_has_cover = track.get("cover", None)  # Yes / No
            file_general_cover_description = track.get(
                "cover_description", None
            )  # Cover
            file_general_cover_type = track.get("cover_type", None)  # Cover (front)
            file_general_cover_mime = track.get("cover_mime", None)  # image/png

        if track.get("track_type") == "Audio":

            ###########################################
            # AUDIO STREAM SPECIFIC DATA
            file_audio_commercial_name = track.get(
                "commercial_name", None
            )  # MPEG Audio
            file_audio_format_version = track.get("format_version", None)  # Version 1
            file_audio_format_profile = track.get("format_profile", None)  # Layer 3
            file_audio_total_duration_in_milliseconds = track.get(
                "duration", None
            )  # 29388
            audio_other_duration_list = track.get("other_duration", None)
            file_audio_total_duration_timestamp = audio_other_duration_list[
                4
            ]  # 00:00:29.388

            file_audio_bit_rate_mode = track.get("bit_rate_mode", None)  # CBR
            audio_other_bit_rate_mode_list = track.get("other_bit_rate_mode", [])
            if len(audio_other_bit_rate_mode_list) > 0:
                file_audio_bit_rate_mode_pretty = audio_other_bit_rate_mode_list[
                    0
                ]  # Constant

            file_audio_bit_rate = track.get("bit_rate", None)  # 128000
            audio_other_bit_rate_list = track.get("other_bit_rate", [])
            if len(audio_other_bit_rate_list) > 0:
                file_audio_bit_rate_pretty = audio_other_bit_rate_list[0]  # 128 kb/s

            file_audio_channel_s = track.get("channel_s", None)  # 1
            audio_other_channel_s_list = track.get("other_channel_s", [])
            if len(audio_other_channel_s_list) > 0:
                file_audio_channel_s_pretty = audio_other_channel_s_list[0]  # 1 channel

            file_audio_samples_per_frame = track.get("samples_per_frame", None)  # 1152

            file_audio_sampling_rate = track.get("sampling_rate", None)  # 44100
            audio_other_sampling_rate_list = track.get("other_sampling_rate", [])
            if len(audio_other_sampling_rate_list) > 0:
                file_audio_sampling_rate_pretty = audio_other_sampling_rate_list[
                    0
                ]  # 44.1 kHz

            file_audio_samples_count = track.get("samples_count", None)  # 1296000

            file_audio_frame_rate = track.get("frame_rate", None)  # 38.281
            audio_other_frame_rate_list = track.get("other_frame_rate", [])
            if len(audio_other_frame_rate_list) > 0:
                file_audio_frame_rate_pretty = audio_other_frame_rate_list[
                    0
                ]  # 38.281 FPS (1152 SPF)

            file_audio_frame_count = track.get("frame_count", None)  # 1125
            file_audio_compression_mode = track.get("compression_mode", None)  # Lossy

            file_audio_stream_size_in_bytes = track.get("stream_size", None)  # 470203
            audio_other_stream_size_list = track.get("other_stream_size", [])
            if len(audio_other_stream_size_list) > 0:
                file_audio_stream_size_pretty = audio_other_stream_size_list[4]  # 459.2 KiB

            file_audio_proportion_of_this_stream = track.get(
                "proportion_of_this_stream", None
            )  # 0.66890

        if track.get("track_type") == "Image":

            ###########################################
            # IMAGE STREAM SPECIFIC DATA

            file_image_format_info = track.get(
                "format_info", None
            )  # Portable Network Graphic
            file_image_commercial_name = track.get("commercial_name", None)  # PNG
            file_image_compression = track.get("compression", None)  # Deflate
            file_image_format_settings = track.get("format_settings", None)  # Linear
            file_image_internet_media_type = track.get(
                "internet_media_type", None
            )  # image/png
            file_image_width = track.get("width", None)  # 3200
            file_image_height = track.get("height", None)  # 3200
            file_image_pixel_aspect_ratio = track.get(
                "pixel_aspect_ratio", None
            )  # 1.000
            file_image_display_aspect_ratio = track.get(
                "display_aspect_ratio", None
            )  # 1.000
            file_image_color_space = track.get("color_space", None)  # RGB

            file_image_bit_depth = track.get("bit_depth", None)  # 8
            image_other_bit_depth_list = track.get("other_bit_depth", [])
            if len(image_other_bit_depth_list) > 0:
                file_image_bit_depth_pretty = image_other_bit_depth_list[0]  # 8 bits

            file_image_compression_mode = track.get(
                "compression_mode", None
            )  # Lossless

            file_image_stream_size_in_bytes = track.get("stream_size", None)  # 230064
            image_other_stream_size = track.get("other_stream_size", [])
            if len(image_other_stream_size) > 0:
                file_image_stream_size_pretty = image_other_stream_size[4]  # 224.7 KiB

            file_image_proportion_of_this_stream = track.get(
                "proportion_of_this_stream", None
            )  # 0.32729

    utc_offset, matched_tz, tz_friendly_name = get_utc_offset_and_us_timezone(
        file_general_file_creation_date__local, file_general_file_creation_date_utc
    )

    full_recording_date_and_time = format_full_datetime(
        file_general_recorded_date_utc, tz_friendly_name
    )

    transcript_dict = {
        "file_general_lyrics": file_general_lyrics,
        "file_general_original_filename": file_general_original_filename,
    }

    random_raw_values = {
        "file_image_bit_depth": file_image_bit_depth,
        "file_image_stream_size_in_bytes": file_image_stream_size_in_bytes,
        "file_image_internet_media_type": file_image_internet_media_type,
        "file_image_commercial_name": file_image_commercial_name,
        "file_audio_stream_size_in_bytes": file_audio_stream_size_in_bytes,
        "file_audio_frame_rate": file_audio_frame_rate,
        "file_audio_sampling_rate": file_audio_sampling_rate,
        "file_audio_channel_s": file_audio_channel_s,
        "file_audio_bit_rate": file_audio_bit_rate,
        "file_audio_bit_rate_mode_pretty": file_audio_bit_rate_mode_pretty,
        "file_audio_total_duration_in_milliseconds": file_audio_total_duration_in_milliseconds,
        "file_general_total_file_size_in_bytes": file_general_total_file_size_in_bytes,
        "file_general_total_duration_in_milliseconds": file_general_total_duration_in_milliseconds,
        "file_general_overall_bitrate": file_general_overall_bitrate,
        "file_general_stream_size_in_bytes": file_general_stream_size_in_bytes,
        "file_general_stream_size_pretty": file_general_stream_size_pretty,
        "file_general_proportion_of_this_stream": file_general_proportion_of_this_stream,
    }

    time_info_dict = {
        "file_general_tagged_date_utc": file_general_tagged_date_utc,
        "file_general_file_creation_date__local": file_general_file_creation_date__local,
        "file_general_file_creation_date_utc": file_general_file_creation_date_utc,
        "file_general_file_last_modification_date__local": file_general_file_last_modification_date__local,
        "file_gerneral_file_last_modification_date_utc": file_gerneral_file_last_modification_date_utc,
        "utc_offset": utc_offset,
        "matched_tz": matched_tz,
    }

    image_info_dict = {
        "file_general_has_cover": file_general_has_cover,
        "file_general_cover_type": file_general_cover_type,
        "file_image_format_info": file_image_format_info,
        "file_general_cover_mime": file_general_cover_mime,
        "file_image_compression_mode": file_image_compression_mode,
        "file_image_compression": file_image_compression,
        "file_image_format_settings": file_image_format_settings,
        "file_image_pixel_aspect_ratio": file_image_pixel_aspect_ratio,
        "file_image_display_aspect_ratio": file_image_display_aspect_ratio,
        "file_image_width": file_image_width,
        "file_image_height": file_image_height,
        "file_image_color_space": file_image_color_space,
        "file_image_bit_depth_pretty": file_image_bit_depth_pretty,
        "file_image_stream_size_pretty": file_image_stream_size_pretty,
        "file_image_proportion_of_this_stream": file_image_proportion_of_this_stream,
    }

    audio_info_dict = {
        "file_audio_commercial_name": file_audio_commercial_name,
        "file_audio_format_version": file_audio_format_version,
        "file_audio_format_profile": file_audio_format_profile,
        "file_audio_compression_mode": file_audio_compression_mode,
        "file_general_track_writing_library": file_general_track_writing_library,
        "file_audio_total_duration_timestamp": file_audio_total_duration_timestamp,
        "file_audio_bit_rate_pretty": file_audio_bit_rate_pretty,
        "file_audio_sampling_rate_pretty": file_audio_sampling_rate_pretty,
        "file_audio_bit_rate_mode": file_audio_bit_rate_mode,
        "file_audio_channel_s_pretty": file_audio_channel_s_pretty,
        "file_audio_frame_rate_pretty": file_audio_frame_rate_pretty,
        "file_audio_frame_count": file_audio_frame_count,
        "file_audio_samples_per_frame": file_audio_samples_per_frame,
        "file_audio_samples_count": file_audio_samples_count,
        "file_audio_stream_size_pretty": file_audio_stream_size_pretty,
        "file_audio_proportion_of_this_stream": file_audio_proportion_of_this_stream,
    }

    track_dict = {
        "file_general_track_album_performer": file_general_track_album_performer,
        "file_general_track_genre": file_general_track_genre,
        "file_general_track_grouping": file_general_track_grouping,
        "file_general_track_more": file_general_track_more,
        "file_general_track_album": file_general_track_album,
        "file_general_track_title": file_general_track_title,
        "track_position_of_total": f"{file_general_track_name_position} / {file_general_track_name_total}",
        "file_general_track_name_position": file_general_track_name_position,
        "file_general_track_name_total": file_general_track_name_total,
    }

    audio_file_metadata = {
        "audio_file_metadata":{
            "file_general_file_name_extension": file_general_file_name_extension,
            # "file_general_recorded_date_utc": file_general_recorded_date_utc,
            "full_recording_date_and_time": full_recording_date_and_time,
            "tz_friendly_name": tz_friendly_name,
            "file_general_internet_media_type": file_general_internet_media_type,
            "file_general_audio_codec": file_general_audio_codec,
            "file_general_number_of_audio_streams": file_general_number_of_audio_streams,
            "file_general_image_codec": file_general_image_codec,
            "file_general_number_of_image_streams": file_general_number_of_image_streams,
            "file_general_overall_bitrate_pretty": file_general_overall_bitrate_pretty,
            "file_general_total_duration_timestamp": file_general_total_duration_timestamp,
            "file_general_total_file_size_pretty": file_general_total_file_size_pretty,
            "file_general_folder_name": file_general_folder_name,
            "file_general_complete_name": file_general_complete_name,
            "track_info": track_dict,
            "audio_info": audio_info_dict,
            "image_info": image_info_dict,
            "time_info": time_info_dict,
            "random_raw": random_raw_values,
            "transcript": transcript_dict,
            "sha256_hash": sha256_hash,
        }
    }

    return audio_file_metadata, media_info
#####################################################################################################################################

#####################################################################################################################################
@logger.catch
def generate_audio_metadata(file_path):
    logger.info(f"Processing: {file_path.name}")

    sha256_hash = compute_sha256(file_path)

    if str(file_path.suffix).strip().lower() == ".mp3":
        formatted_media_info, orig_media_info = extract_mp3_info(file_path, sha256_hash)

    if str(file_path.suffix).strip().lower() == ".flac":
        formatted_media_info, orig_media_info = extract_mp3_info(file_path, sha256_hash)
    
    return formatted_media_info, orig_media_info
#####################################################################################################################################


#####################################################################################################################################
def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__str__"):
        return str(obj)
    else:
        return obj

#####################################################################################################################################