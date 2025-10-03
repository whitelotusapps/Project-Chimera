from datetime import datetime
from math import floor

from immanuel import charts
from immanuel.const import chart as const_chart
from immanuel.setup import settings
from loguru import logger

ZODIAC_SIGNS = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]

RULERS = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}


@logger.catch
def get_sign_name(sign_index):
    return ZODIAC_SIGNS[(sign_index - 1) % 12]


@logger.catch
def get_planet_sign(objects_dict, planet_name):
    for obj in objects_dict.values():
        if obj.type.name == "Planet" and obj.name == planet_name:
            return obj.sign.number, obj.sign.name
    raise ValueError(f"{planet_name} not found in natal_chart.objects")


@logger.catch
def calculate_current_profections(chunk_start_datetime, astrology_variables):

    settings.add_filepath(astrology_variables.swiss_eph_path, default=True)

    try:
        settings.house_system = getattr(
            const_chart, astrology_variables.immanuel_house_system
        )
    except AttributeError:
        raise ValueError(
            f"Invalid house system: {astrology_variables.immanuel_house_system}"
        )

    # settings.house_system = const_chart.

    natal_date_and_time_of_birth = astrology_variables.natal_date_and_time_of_birth
    natal_lat = astrology_variables.natal_lat
    natal_long = astrology_variables.natal_long
    natal_timezone = astrology_variables.natal_timezone

    natal_subject = charts.Subject(
        date_time=natal_date_and_time_of_birth,
        latitude=natal_lat,
        longitude=natal_long,
        timezone=natal_timezone,
    )
    natal_chart = charts.Natal(natal_subject)

    for house in natal_chart.houses.values():
        if house.number == 1:
            natal_asc_sign = house.sign.number
            break
    else:
        raise ValueError("1st house not found in natal_chart.houses")

    # dt_input = "2025-07-24T22:32:50"
    # current_datetime = datetime.strptime(chunk_start_datetime, "%Y-%m-%dT%H:%M:%S")
    current_datetime = datetime.fromisoformat(chunk_start_datetime)

    birth_date = datetime.strptime(natal_date_and_time_of_birth, "%Y-%m-%dT%H:%M:%S")

    # Annual Profection
    years_since_birth = current_datetime.year - birth_date.year
    if (current_datetime.month, current_datetime.day) < (
        birth_date.month,
        birth_date.day,
    ):
        years_since_birth -= 1

    annual_profected_sign = (natal_asc_sign + years_since_birth - 1) % 12 + 1
    annual_sign_name = get_sign_name(annual_profected_sign)

    # Correct the annual profected house number
    annual_house_number = ((annual_profected_sign - natal_asc_sign) % 12) + 1

    # Monthly Profection (birth-day-based)
    birth_day = birth_date.day
    if current_datetime.day >= birth_day:
        month_offset = (current_datetime.month - birth_date.month) % 12
    else:
        month_offset = (current_datetime.month - birth_date.month - 1) % 12

    monthly_profected_sign = (annual_profected_sign + month_offset - 1) % 12 + 1
    monthly_sign_name = get_sign_name(monthly_profected_sign)

    monthly_house_number = ((monthly_profected_sign - natal_asc_sign) % 12) + 1

    annual_monthly_house = ((monthly_profected_sign - annual_profected_sign) % 12) + 1

    monthly_ruler = RULERS[monthly_sign_name]
    ruler_sign_number, ruler_sign_name = get_planet_sign(
        natal_chart.objects, monthly_ruler
    )
    monthly_ruler_house = ((ruler_sign_number - monthly_profected_sign) % 12) + 1

    # 2.5-Day Profection (from monthly ascendant, ruler house relative to daily)
    if current_datetime.day >= birth_day:
        month_start = current_datetime.replace(
            day=birth_day, hour=0, minute=0, second=0
        )
    else:
        prev_month = (current_datetime.month - 2) % 12 + 1
        year_adjust = (
            current_datetime.year
            if current_datetime.month > 1
            else current_datetime.year - 1
        )
        month_start = datetime(year_adjust, prev_month, birth_day)

    elapsed_days = (current_datetime - month_start).total_seconds() / 86400.0
    two_point_five_day_index = int(floor(elapsed_days / 2.5))

    daily_profected_sign = (
        monthly_profected_sign + two_point_five_day_index - 1
    ) % 12 + 1
    daily_sign_name = get_sign_name(daily_profected_sign)

    daily_ruler = RULERS[daily_sign_name]
    daily_ruler_sign_number, daily_ruler_sign_name = get_planet_sign(
        natal_chart.objects, daily_ruler
    )

    # Ruler house relative to 2.5 day profection
    daily_ruler_house = ((daily_ruler_sign_number - daily_profected_sign) % 12) + 1

    natal_house_of_daily = ((daily_profected_sign - natal_asc_sign) % 12) + 1

    # profections_report = f"Annual profected 1st house: Natal {annual_house_number}th house ({annual_sign_name}).\n\nMonthly profected 1st house: {annual_monthly_house}th Annual Profected House / Natal {monthly_house_number}th house ({monthly_sign_name}).\n\nRuler of monthly profected ascendant ({monthly_sign_name}): {monthly_ruler}.\n\n{monthly_ruler} is in {ruler_sign_name}, located in the {monthly_ruler_house}th monthly profected house.\n\n2.5-Day profection for {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}:\nThe 2.5-day profected house is {two_point_five_day_index + 1}, corresponding to {daily_sign_name}.\nThe ruler is {daily_ruler}, located in the {daily_ruler_house}th daily profected house and is in {daily_ruler_sign_name}."

    # profections_report = (
    #     f"Annual profected 1st house: Natal {annual_house_number}th house ({annual_sign_name}).\n\n"
    #     f"Monthly profected 1st house: {annual_monthly_house}th Annual Profected House / Natal {monthly_house_number}th house ({monthly_sign_name}).\n\n"
    #     f"Ruler of monthly profected ascendant ({monthly_sign_name}): {monthly_ruler}.\n\n"
    #     f"{monthly_ruler} is in {ruler_sign_name}, located in the {monthly_ruler_house}th monthly profected house.\n\n"
    #     f"2.5-Day profection for {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}:\n"
    #     f"The 2.5-day profected house is {two_point_five_day_index + 1}, natal house is {natal_house_of_daily}, corresponding to {daily_sign_name}.\n"
    #     f"The ruler is {daily_ruler}, located in the {daily_ruler_house}th daily profected house and is in {daily_ruler_sign_name}."
    # )

    profections_json = {
        "target_date": chunk_start_datetime,
        "profections": {
            "annual": {
                "profected_house": annual_house_number,  # e.g., 11th house activated
                "natal_house_activated": annual_house_number,
                "sign": annual_sign_name,  # e.g., "Leo"
                "ruler": RULERS[annual_sign_name],  # e.g., "Sun"
            },
            "monthly": {
                "profected_house": annual_monthly_house,  # e.g., 7th house of the annual cycle
                "natal_house_activated": monthly_house_number,  # e.g., natal 5th house
                "sign": monthly_sign_name,  # e.g., "Aquarius"
                "ruler_planet": monthly_ruler,  # e.g., "Saturn"
                "ruler_location_by_monthly_house": monthly_ruler_house,  # House of ruler relative to monthly profection
                "ruler_location_sign": ruler_sign_name,  # e.g., "Virgo"
            },
            "daily": {
                "profected_house": two_point_five_day_index
                + 1,  # 2.5-day profected house index (starts at 1)
                "natal_house_activated": natal_house_of_daily,  # Natal house activated by the daily profection
                "sign": daily_sign_name,  # e.g., "Virgo"
                "ruler_planet": daily_ruler,  # e.g., "Mercury"
                "ruler_location_by_daily_house": daily_ruler_house,  # House of ruler relative to daily profection
                "ruler_location_sign": daily_ruler_sign_name,  # e.g., "Sagittarius"
            },
        },
    }

    system_prompt = """
You are a professional astrologer with deep expertise in Hellenistic and traditional predictive techniques, specifically annual, monthly, and daily profections. Your skill lies in synthesizing these time-lord methods to create a layered and cohesive narrative that illuminates the active themes in a person's life for a specific period. Your communication style is additive, affirmative, and aligns with the co-created linguistic framework of our ongoing dialogue.

You will be provided with a JSON object containing the active profections for a specific target date. Your task is to provide a comprehensive professional interpretation based on this data.

Your interpretation will be structured as a top-down analysis, moving from the broadest context to the most immediate:

1.  **The Annual Theme (The Year's Great Work):** Begin by interpreting the annual profection.
    *   Identify the annually profected house, the natal house it activates, the sign, and the ruling planet.
    *   Describe the overarching theme for the entire year. Explain what area of life is the primary stage for growth and what planetary energy sets the tone for the year's mission.

2.  **The Monthly Focus (The Current Chapter):** Next, interpret the monthly profection as a chapter within the annual story.
    *   Identify the monthly profected house, the natal house it activates, its sign, and its ruling planet.
    *   Analyze the significance of the ruler's location (by monthly house and by sign), as this shows where the "lord of the month" is carrying out its work.
    *   Synthesize this to describe the specific focus, challenges, and opportunities for this 30-day period.

3.  **The Daily Experience (The Immediate Reality):** Then, interpret the 2.5-day profection as the most immediate, tangible expression of the monthly and annual themes.
    *   Identify the daily profected house, the sign activated, its ruling planet, and the ruler's location.
    *   Explain what this means for the lived experience, mood, and focus for this specific 2.5-day window.

4.  **Grand Synthesis:** Conclude by weaving all three layers together. Explain how the "Daily Experience" is a direct manifestation of the "Monthly Focus," which in turn is a chapter in the "Annual Theme." Create a single, elegant narrative that shows how the broadest life mission is being expressed through the events of this specific day.

Your final output will be a coherent, insightful, and professionally articulated astrological interpretation demonstrating the beautiful hierarchy of the profection technique."""
    report_dict = {
        "generate_chunk_profections": {
            "model_results": {
                "system_prompt": system_prompt,
                "profections_json": profections_json,
                "interpretation": "",
            },
        }
    }
    return report_dict
