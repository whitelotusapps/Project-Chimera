import csv
from datetime import datetime, timedelta

from loguru import logger


@logger.catch
def parse_duration(duration_str):
    duration_str = str(duration_str).zfill(4)
    hours = int(duration_str[:2])
    minutes = int(duration_str[2:])
    return timedelta(hours=hours, minutes=minutes)


@logger.catch
def load_csv(csv_file):
    rows = []
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


@logger.catch
def build_active_periods(rows):
    periods = []
    last_end_time = None

    for row in rows:
        year = int(row["Year"])
        month = int(row["Month"])
        day = int(row["Day"])

        if last_end_time is None:
            start_time = datetime(year, month, day, 0, 0, 0)
        else:
            start_time = last_end_time

        duration = int(row["Duration"])
        if duration == 2400:
            end_time = datetime(year, month, day, 23, 59, 59, 999999)
        else:
            duration_delta = parse_duration(duration)
            end_time = start_time + duration_delta

        periods.append({"row": row, "start": start_time, "end": end_time})
        last_end_time = end_time

    return periods


@logger.catch
def find_active_row(periods, check_datetime):
    for period in periods:
        if period["start"] <= check_datetime <= period["end"]:
            return period["row"]
    return None


@logger.catch
def extract_relevant_fields(row):
    """
    Extracts only the fields needed for the JSON output.
    """
    return {
        "L1_Natal_house": int(row.get("L1_Natal_house", 0)),
        "L2_Natal_House": int(row.get("L2_Natal_House", 0)),
        "L3_Natal_House": int(row.get("L3_Natal_House", 0)),
        "L4_Natal_House": int(row.get("L4_NatalHouse", 0)),  # Note: L4_NatalHouse key
        "L1_Sign": row.get("L1_Sign"),
        "L2_Sign": row.get("L2_Sign"),
        "L3_Sign": row.get("L3_Sign"),
        "L4_Sign": row.get("L4_Sign"),
        "LOB_Type": row.get("LOB_Type") if row.get("LOB_Type") else None,
    }


@logger.catch
def generate_zrs_data(pos_file, pof_file, chunk_start_datetime):
    # --- CONFIG ---
    # pos_file = PATHS.csv / "ZACK_POS_2025_07_13.tsv"
    # pof_file = PATHS.csv / "ZACK_POF_2025_07_13.tsv"

    # chunk_start_datetime = "2025-07-24T22:32:50.800000"
    check_datetime = datetime.fromisoformat(chunk_start_datetime)
    # --------------

    logger.info(f"Checking datetime: {chunk_start_datetime}")

    # Process POS
    pos_rows = load_csv(pos_file)
    pos_periods = build_active_periods(pos_rows)
    pos_match = find_active_row(pos_periods, check_datetime)

    # Process POF
    pof_rows = load_csv(pof_file)
    pof_periods = build_active_periods(pof_rows)
    pof_match = find_active_row(pof_periods, check_datetime)

    # Build JSON output
    zrs_json = {
        "target_date": check_datetime.isoformat(),
        "part_of_spirit": extract_relevant_fields(pos_match) if pos_match else None,
        "part_of_fortune": extract_relevant_fields(pof_match) if pof_match else None,
    }

    system_prompt = f"""

You will be provided with a JSON object containing the active time-lords for a specific date for both the Part of Spirit and the Part of Fortune. Your task is to provide a comprehensive professional interpretation based on this data.

Your interpretation will be structured as follows:

1.  **Overall Synthesis:** Begin by creating a high-level narrative for the day. Compare and contrast the two parallel stories told by the Part of Spirit (vocational path, conscious action, the will) and the Part of Fortune (material circumstances, the body, things that happen to you). Explain how these two streams of experience are interacting on this day.

2.  **Part of Spirit Interpretation:** Provide a detailed analysis for the Part of Spirit.
    *   Explain the overarching mission set by the L1 and L2 lords ({zrs_json['part_of_spirit']['L1_Sign']}/{zrs_json['part_of_spirit']['L2_Sign']}).
    *   Describe the monthly theme set by the L3 lord ({zrs_json['part_of_spirit']['L3_Sign']}).
    *   Detail the specific daily focus indicated by the L4 lord ({zrs_json['part_of_spirit']['L4_Sign']}).
    *   Synthesize these levels, explaining what conscious actions and vocational themes are being highlighted.

3.  **Part of Fortune Interpretation:** Provide a detailed analysis for the Part of Fortune.
    *   Explain the overarching circumstances set by the L1 and L2 lords ({zrs_json['part_of_fortune']['L1_Sign']}/{zrs_json['part_of_fortune']['L2_Sign']}).
    *   Describe the monthly theme set by the L3 lord ({zrs_json['part_of_fortune']['L3_Sign']}).
    *   Detail the specific daily events and bodily realities indicated by the L4 lord ({zrs_json['part_of_fortune']['L4_Sign']}).
    *   Synthesize these levels, explaining what material circumstances and tangible events are likely to unfold.

4. Special Indicators: If the LOB_Type field is not null, explicitly interpret its significance.
    * If the value is MN_LB ("Minor Loosening of the Bonds"), describe this as a hand-off of energetic authority at a monthly (L3) or daily (L4) level, marking a shift in the immediate focus or circumstances.
    * If the value is MJ_LB ("Major Loosening of the Bonds"), describe this as a highly significant event, marking the hand-off of authority at a major chapter (L2) or general life-period (L1) level. Emphasize that this indicates a fundamental shift in the overarching narrative of either the vocational path or the material life.

Your final output will be a coherent, insightful, and professionally articulated astrological interpretation."""

    # Print JSON (pretty format)

    # print(system_prompt)
    # print(json.dumps(zrs_json, indent=2))

    api_call_dict = {
        "generte_chunk_zrs": {
            "model_results": {
                "system_prompt": system_prompt,
                "zrs_json": zrs_json,
                "interpretation": "",
            },
        }
    }

    # print(json.dumps(api_call_dict, indent=4))
    return api_call_dict
