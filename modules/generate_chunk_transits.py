# THIS FILE CALCULATES THE TRANSITS BETWEEN MY NATAL CHART AND A SPECIFIED DATE AND TIME OF THE TRANSITS
# THIS CODE IS BASED ON THE AUTHOR OF IMMANUEL'S SUGGESTED USE
# SEE THIS CHAT:
# https://chatgpt.com/c/686d3ab0-1f28-800c-9222-bf91217dd70d

import os

from immanuel import charts
from immanuel.const import calc
from immanuel.const import chart as const_chart
from immanuel.reports import aspect
from immanuel.setup import settings

# Set your local ephemeris path
settings.add_filepath("C:\\temp\\code\\astrology\\eph", default=True)

os.system("clear")
#####################################################################################################################################
# Set house system to Whole Sign
settings.house_system = const_chart.WHOLE_SIGN

# Assuming you want all aspects to/from these points
points_to_enable = [
    const_chart.ASC,
    const_chart.DESC,
    const_chart.MC,
    const_chart.IC,
    const_chart.TRUE_NORTH_NODE,
    const_chart.TRUE_SOUTH_NODE,
    const_chart.VERTEX,
    const_chart.PART_OF_FORTUNE,
    const_chart.TRUE_LILITH,
    const_chart.SUN,
    const_chart.MOON,
    const_chart.MERCURY,
    const_chart.VENUS,
    const_chart.MARS,
    const_chart.JUPITER,
    const_chart.SATURN,
    const_chart.URANUS,
    const_chart.NEPTUNE,
    const_chart.PLUTO,
    const_chart.CHIRON,
    const_chart.PALLAS,
    const_chart.JUNO,
    const_chart.CERES,
    const_chart.VESTA,
]

# Define which aspects they can initiate and receive
all_aspects = [
    calc.CONJUNCTION,
    calc.OPPOSITION,
    calc.SQUARE,
    calc.TRINE,
    calc.SEXTILE,
    calc.SEMISEXTILE,
    calc.QUINCUNX,
    calc.QUINTILE,
    calc.BIQUINTILE,
    calc.SEPTILE,
    calc.SESQUISQUARE,
    calc.SEMISQUARE,
]

# Apply to settings
# for point in points_to_enable:
#     settings.aspect_rules[point] = {
#         "initiate": all_aspects,
#         "receive": all_aspects,
#     }

settings.default_aspect_rule = {
    "initiate": all_aspects,
    "receive": all_aspects,
}


#####################################################################################################################################


#####################################################################################################################################
def obj_to_dict(obj):
    return {
        "index": obj.index,
        "lon": obj.longitude.degrees,  # Convert Angle -> float degrees
        "speed": obj.speed,
        "name": obj.name,
    }


#####################################################################################################################################
def extract_nested_attr(obj, key_path):
    """
    Recursively extract a nested attribute from an object or dictionary.

    :param obj: object or dict to traverse
    :param key_path: list of attribute names
    :return: value or None
    """
    for key in key_path:
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
        if obj is None:
            return None
    return obj


#####################################################################################################################################
def find_item(data_dict, search_value, search_by="name"):
    """
    Generic lookup for an item in a dictionary of wrap.Objects.

    :param data_dict: dict of wrap.Object instances (objects or houses)
    :param search_value: the lookup value
    :param search_by: 'name', 'number', or 'sign'
    :return: wrap.Object or None
    """
    for item in data_dict.values():
        if search_by == "name" and getattr(item, "name", None) == search_value:
            return item
        elif search_by == "number" and getattr(item, "number", None) == search_value:
            return item
        elif search_by == "sign":
            sign_obj = getattr(item, "sign", None)
            if sign_obj and getattr(sign_obj, "name", None) == search_value:
                return item
    return None


#####################################################################################################################################
def extract_values(data_dict, query_list, search_by="name"):
    """
    Extract values from a data dictionary (objects or houses) for given points.

    :param data_dict: .objects or .houses dict from a chart
    :param query_list: list like:
        [{"point": "Venus", "keys": [["house", "name"], ["sign", "name"]]}]
    :param search_by: 'name', 'number', or 'sign'
    :return: dict of results
    """
    results = {}

    for query in query_list:
        point = query.get("point")
        key_paths = query.get("keys", [])

        item = find_item(data_dict, point, search_by)
        if not item:
            results[point] = None
            continue

        results[point] = {}

        for path in key_paths:
            label = ".".join(path)
            value = extract_nested_attr(item, path)
            results[point][label] = value

        # Include the item number if it exists
        results[point]["item_number"] = getattr(item, "number", None)

    return results


#####################################################################################################################################
def find_aspects_between_charts(base_chart, from_chart, orb=2.0, aspect_types=None):

    settings.planet_orbs = {
        calc.CONJUNCTION: orb,
        calc.OPPOSITION: orb,
        calc.SQUARE: orb,
        calc.TRINE: orb,
        calc.SEXTILE: orb,
        calc.SEPTILE: orb,
        calc.SEMISQUARE: orb,
        calc.SESQUISQUARE: orb,
        calc.SEMISEXTILE: orb,
        calc.QUINCUNX: orb,
        calc.QUINTILE: orb,
        calc.BIQUINTILE: orb,
    }

    if aspect_types is None:
        aspect_types = [
            calc.CONJUNCTION,
            calc.OPPOSITION,
            calc.SQUARE,
            calc.TRINE,
            calc.SEXTILE,
        ]

    base_objects = base_chart.objects
    from_objects = from_chart.objects
    found_aspects = []

    for obj_from in from_objects.values():
        dict_from = obj_to_dict(obj_from)
        for obj_base in base_objects.values():
            dict_base = obj_to_dict(obj_base)
            asp = aspect.between(dict_from, dict_base)
            # print(f"{json.dumps(asp, indent=4)}")
            # input("\n\nHERE\n\n")
            if asp and asp["aspect"] in aspect_types and abs(asp["difference"]) <= orb:
                # if asp and asp["aspect"] in aspect_types and abs(asp["difference"]):
                found_aspects.append(
                    {
                        "object": obj_from.name,
                        "to_object": obj_base.name,
                        "aspect_type": asp["aspect"],
                        "orb": asp["difference"],
                    }
                )

    return found_aspects


#####################################################################################################################################
def calculate_chunk_transits(chunk_calendar_start_datetime, orb):
    # Your natal and transit chart setup here:
    natal_subject = charts.Subject(
        date_time="1978-12-15 01:24:00",
        latitude="38n58'56''",
        longitude="094w40'14''",
        timezone="America/Chicago",
    )

    natal_chart = charts.Natal(natal_subject)

    transit_subject = charts.Subject(
        date_time=chunk_calendar_start_datetime,
        latitude="38n58'56''",
        longitude="094w40'14''",
        timezone="America/Chicago",
    )

    transit_chart = charts.Natal(
        transit_subject
    )  # Transit chart is a natal chart at transit time

    # Assuming you already have natal_chart and transit_chart created:
    natal_chart = charts.Natal(natal_subject)
    transit_chart = charts.Natal(transit_subject)

    natal_chart_objects = natal_chart.objects
    transit_chart_objects = transit_chart.objects

    natal_chart_houses = natal_chart.houses

    # Find aspects
    aspects_found = find_aspects_between_charts(natal_chart, transit_chart, orb)

    # Print nicely
    aspect_names = {
        calc.CONJUNCTION: "Conjunction",
        calc.OPPOSITION: "Opposition",
        calc.SQUARE: "Square",
        calc.TRINE: "Trine",
        calc.SEXTILE: "Sextile",
    }

    all_transits = []
    all_transits_text = []

    for asp in aspects_found:

        natal_point = asp["to_object"]
        transit_point = asp["object"]

        # Example input query
        natal_point_house_sign_list = [
            {"point": natal_point, "keys": [["house", "name"], ["sign", "name"]]},
        ]

        ##################################################
        # LOOK UP NATAL VALUES
        natal_point_result = extract_values(
            natal_chart_objects, natal_point_house_sign_list
        )

        natal_point_house_name = natal_point_result.get(natal_point, {}).get(
            "house.name", ""
        )
        natal_point_sign_name = natal_point_result.get(natal_point, {}).get(
            "sign.name", ""
        )

        ##################################################
        # LOOK UP WHAT SIGN THE TRANSITING PLANET IS IN
        transit_point_house_sign_list = [
            {"point": transit_point, "keys": [["sign", "name"]]},
        ]

        transit_point_result = extract_values(
            transit_chart_objects, transit_point_house_sign_list
        )

        transit_point_sign_name = transit_point_result.get(transit_point, {}).get(
            "sign.name", ""
        )

        ##################################################
        # DO A "REVERSE LOOK UP" BECAUSE WE NEED TO FIND OUT WHICH **NATAL** HOUSE THE TRANSITING PLANET IS IN
        # AND WE DO THIS BASED UPON WHAT SIGN THE TRANSITING PLANET IS IN
        transit_point_house_sign_list = [
            {"point": transit_point_sign_name, "keys": [["name"]]},
        ]

        # WE HAVE TO PASS THE NATAL CHART HOUSES OBJECT SO WE CAN FIND OUT THE HOUSE NUMBER FOR THE SIGN THAT THE TRANSITING
        # PLANET IS IN
        transit_point_result = extract_values(
            natal_chart_houses, transit_point_house_sign_list, search_by="sign"
        )

        # GET THE NAME OF THE HOUSE THAT HAS THE SIGN THAT THTE TRANSITING PLANET IS IN
        transit_point_house_name = transit_point_result.get(
            transit_point_sign_name, {}
        ).get("name", "")
        ##################################################

        aspect_name = aspect_names.get(asp["aspect_type"], str(asp["aspect_type"]))

        # aspect_text = f"{transit_point} {aspect_name} {natal_point} within {asp['orb']} degree(s) orb. Transiting {transit_point} is in the {transit_point_house_name} and is in {transit_point_sign_name}. Natal {natal_point} is in the {natal_point_house_name} and is in {natal_point_sign_name}."

        aspect_text = f"Transiting {transit_point} {aspect_name} Natal {natal_point}. Transiting {transit_point} is in the {transit_point_house_name} and is in {transit_point_sign_name}. Natal {natal_point} is in the {natal_point_house_name} and is in {natal_point_sign_name}."

        if "South Node" not in aspect_text:
            aspect_orb = asp["orb"]

            transit_dict = {
                "transit_point": transit_point,
                "transit_point_house": transit_point_house_name,
                "transit_point_sign": transit_point_sign_name,
                "aspect_type": aspect_name,
                "natal_point": natal_point,
                "natal_point_house": natal_point_house_name,
                "natal_point_sign": natal_point_sign_name,
                "orb": aspect_orb,
                "text": aspect_text,
            }
            all_transits.append(transit_dict)
            all_transits_text.append(aspect_text)

    final_transit_dict = {
        "generate_chunk_transits": {
            "model_results": {
                "datetime_of_transits": chunk_calendar_start_datetime,
                "transits": all_transits,
                "number_of_transits": len(all_transits_text),
                "all_transits_text": all_transits_text,
            },
        }
    }

    return final_transit_dict
