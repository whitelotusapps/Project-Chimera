from collections import Counter

import matplotlib.colors as mcolors
import pandas as pd
import stylecloud
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import STOPWORDS

from .project_paths import PATHS


#####################################################################################################################################
# Load the valence CSV file
@logger.catch
def load_valence_data(csv_file):
    """
    Load valence, arousal, and dominance data from a CSV file into a pandas DataFrame.

    This function reads a CSV file and extracts specific columns related to
    emotional valence (V), arousal (A), and dominance (D) scores for words.

    Args:
        csv_file (str): Path to the CSV file containing valence data.

    Returns:
        pandas.DataFrame: A DataFrame containing the columns:
            - "Word": The word being measured.
            - "V.Mean.Sum": Valence mean sum score.
            - "A.Mean.Sum": Arousal mean sum score.
            - "D.Mean.Sum": Dominance mean sum score.

    Side Effects:
        - Reads from disk (I/O operation).
        - Raises an exception if the file is not found or is unreadable.

    Notes:
        - Assumes the CSV file contains columns named exactly "Word", "V.Mean.Sum", "A.Mean.Sum", and "D.Mean.Sum".
        - Returned DataFrame only includes these four columns.

    Caveats:
        - Does not perform any validation on the data beyond selecting columns.
        - If any expected column is missing, pandas will raise a `KeyError`.
        - Does not handle file encoding issues; assumes default encoding.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        pandas.errors.ParserError: If the CSV file cannot be parsed.
        KeyError: If any required column is missing from the CSV.
    """
    df = pd.read_csv(csv_file)
    return df[
        ["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
    ]  # Assuming the columns are named "Word" and "V.Mean.Sum"


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def calculate_valence_averages(word_dict):
    """
    Calculate average valence, arousal, and dominance scores for given words.

    This function loads valence data from a CSV file, then looks up each word
    in the provided dictionary within the valence DataFrame. It computes the
    average values for the columns "V.Mean.Sum", "A.Mean.Sum", and "D.Mean.Sum"
    based on matched words.

    Args:
        word_dict (dict): Dictionary with words as keys. The values are not used.

    Returns:
        dict: A dictionary containing the average scores:
            - "V.Mean.Sum": Average valence score, or None if no matches found.
            - "A.Mean.Sum": Average arousal score, or None if no matches found.
            - "D.Mean.Sum": Average dominance score, or None if no matches found.

    Side Effects:
        - Reads valence data from disk (I/O operation) each time it is called.
        - Loads a CSV file from a fixed path relative to the script.

    Notes:
        - Assumes the CSV file at 'assets/CSV/2023-04-05-words.csv' exists and
          is formatted correctly with the expected columns.
        - Only words present in the CSV file contribute to the average calculation.
        - If no words from `word_dict` are found in the CSV, returns averages as None.

    Caveats:
        - Inefficient for large `word_dict` due to filtering the DataFrame repeatedly inside a loop.
        - No normalization or case-insensitive matching is performed on words.
        - Does not handle exceptions if the CSV file is missing or corrupted.
        - The CSV file path is hardcoded; consider parameterizing for flexibility.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pandas.errors.ParserError: If the CSV file cannot be parsed.
    """
    valence_df = load_valence_data(PATHS.csv / "Ratings_Warriner_et_al.csv")
    values = {"V.Mean.Sum": [], "A.Mean.Sum": [], "D.Mean.Sum": []}

    for word in word_dict.keys():
        row = valence_df[valence_df["Word"] == word]
        if not row.empty:
            values["V.Mean.Sum"].append(row["V.Mean.Sum"].values[0])
            values["A.Mean.Sum"].append(row["A.Mean.Sum"].values[0])
            values["D.Mean.Sum"].append(row["D.Mean.Sum"].values[0])

    if not values["V.Mean.Sum"]:  # No words found in CSV
        return {"V.Mean.Sum": None, "A.Mean.Sum": None, "D.Mean.Sum": None}

    return {
        "V.Mean.Sum": sum(values["V.Mean.Sum"]) / len(values["V.Mean.Sum"]),
        "A.Mean.Sum": sum(values["A.Mean.Sum"]) / len(values["A.Mean.Sum"]),
        "D.Mean.Sum": sum(values["D.Mean.Sum"]) / len(values["D.Mean.Sum"]),
    }


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def generate_color_palette(averages):
    """
    Generate a color palette based on emotional component averages.

    This function creates a palette of colors by mapping valence, arousal,
    and dominance averages to HSV color space parameters (hue, saturation,
    and lightness), then converts them to RGB hex codes. The palette is
    designed to reflect emotional tone through color variation and intensity.

    Args:
        averages (dict): Dictionary containing average emotional components:
            - "V.Mean.Sum" (float or None): Valence average score.
            - "A.Mean.Sum" (float or None): Arousal average score.
            - "D.Mean.Sum" (float or None): Dominance average score.

    Returns:
        list: A list of hex color strings representing the generated palette.

    Side Effects:
        - None. This function is purely computational and returns a new list.

    Notes:
        - Emotional averages are expected to be numeric and roughly in a 0-9 range.
          The function uses default fallback values when None or zero values appear.
        - The function dynamically adjusts palette size and color properties based
          on input averages.
        - Colors avoid “rust” tones and emphasize cooler hues like green, blue, and purple.
        - The palette size varies between about 32 and 256 colors depending on arousal.
        - Uses matplotlib's `hsv_to_rgb` and `rgb2hex` for color space conversions.

    Caveats:
        - Input values are not validated for type or range; invalid values may produce
          unexpected colors or runtime errors.
        - The function uses hardcoded hue values and constants, which limits palette
          customization without modifying the code.
        - The gamma adjustment and lightness/saturation calculations are heuristic
          and may not map perfectly to subjective emotional experiences.
        - For extremely low dominance values, colors are reversed to reflect different moods.

    Examples:
        >>> averages = {"V.Mean.Sum": 5.0, "A.Mean.Sum": 6.0, "D.Mean.Sum": 4.0}
        >>> palette = generate_color_palette(averages)
        >>> print(palette[:3])  # prints first 3 hex colors

    """
    # Normalize and adjust based on emotional components
    # Hue: Using the emotional components to span the full color spectrum (0 to 360 degrees)
    start_hue = (averages["V.Mean.Sum"] * 40) - (
        averages["D.Mean.Sum"] * 20
    )  # Valence and Dominance impact
    start_hue = max(0, min(360, start_hue))  # Ensure hue is within the 0-360 range

    end_hue = (averages["A.Mean.Sum"] * 40) + (
        averages["V.Mean.Sum"] * 20
    )  # Arousal and Valence impact
    end_hue = max(0, min(360, end_hue))  # Ensure hue is within the 0-360 range

    # Adjust lightness and saturation based on the emotional averages
    gamma = (
        0.8 + (averages["D.Mean.Sum"] or 5) / 9 * 1.0
    )  # Ensure vividness for high dominance
    sat = (
        0.7 + (averages["A.Mean.Sum"] or 5) / 9 * 0.8
    )  # More saturation for higher arousal
    min_sat = 0.9  # Ensure vibrant colors
    max_sat = 1.0  # Max saturation
    min_light = 0.4  # Avoid too dark colors (low dominance)
    max_light = 0.9  # Keep brightness high for emotional expressiveness
    reverse = (averages["D.Mean.Sum"] or 5) < 4.5  # Reverse colors for lower dominance

    n = int(
        32 + (averages["A.Mean.Sum"] or 5) / 9 * 224
    )  # Increase palette size for diverse emotional tones

    # Modify hues to avoid "rust" tones (red, orange), and introduce cooler tones (blue, green, purple)
    hues = [
        120,  # Green (Calmness, Relaxation)
        180,  # Blue (Sadness, Neutral)
        240,  # Purple (Confusion, Uncertainty)
        300,  # Blue-Purple (Deep reflection)
        60,  # Yellow (Happiness, Joy)
        160,  # Teal (Balance, Serenity)
        330,  # Violet (Nostalgia, Introspection)
    ]

    # Calculate the number of hues and determine how to cycle through them
    hue_steps = max(1, n // len(hues))  # Ensure hue_steps is at least 1
    palette_colors = []
    for i in range(n):
        # Cycle through hues based on hue_steps
        hue = hues[(i // hue_steps) % len(hues)]  # Ensure hue is within range
        # Adjust lightness and saturation dynamically to reflect the emotion's intensity
        lightness = min_light + (max_light - min_light) * (i / n)
        color = mcolors.hsv_to_rgb((hue / 360, sat, lightness))  # Convert HSV to RGB
        palette_colors.append(color)

    # Ensure the RGB values are within [0, 1] range
    normalized_palette_colors = [
        (max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
        for r, g, b in palette_colors
    ]

    # Convert the normalized RGB values into hex color codes
    hex_palette = [mcolors.rgb2hex(c) for c in normalized_palette_colors]

    return hex_palette


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def generate_stylecloud(text_file, output_image, top_n=100, custom_stopwords=None):
    """
    Generate a styled word cloud image from a text file using TF-IDF weighted words and a custom color palette.

    This function reads text from a file, filters stopwords, dynamically removes the most common words,
    applies TF-IDF weighting to identify the top N significant words, computes valence-based color averages,
    generates a color palette, and creates a StyleCloud word cloud image saved to disk.

    Args:
        text_file (str): Path to the input text file.
        output_image (str): Path where the generated word cloud image will be saved.
        top_n (int, optional): Number of top TF-IDF weighted words to include in the word cloud. Defaults to 100.
        custom_stopwords (set or list, optional): Additional stopwords to exclude from the word cloud. Defaults to None.

    Side Effects:
        - Reads from the specified text file.
        - Writes an image file representing the word cloud to the specified output path.
        - Uses logging to record the path where the word cloud is saved.

    Notes:
        - Stopwords are combined from the default STOPWORDS set and any custom stopwords provided.
        - Words are filtered to exclude the most common 10 words dynamically identified in the text.
        - TF-IDF weighting is applied on filtered words to select the most significant words.
        - The valence averages for the selected words are computed and used to generate a color palette.
        - The StyleCloud library is used to generate and save the word cloud image.
        - The font path is hardcoded relative to `script_path` and requires the font file to exist there.

    Caveats:
        - The input text file must exist and be readable, or the function will raise an IOError.
        - The function expects that STOPWORDS and `script_path` are defined in the global scope.
        - If TF-IDF vectorization or StyleCloud generation fails, exceptions will propagate up.
        - The function assumes that the `stylecloud` library is installed and properly configured.
        - Word cloud appearance and size parameters are hardcoded but can be modified within the function.
        - Logging depends on a configured `logger` object in the global scope.

    Example:
        >>> generate_stylecloud("data/sample.txt", "output/cloud.png", top_n=50, custom_stopwords={"example"})
    """
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Define stopwords
    stopwords = (
        STOPWORDS.union(set(custom_stopwords)) if custom_stopwords else STOPWORDS
    )
    stopwords = set(word.lower() for word in stopwords)  # Ensure lowercase consistency

    # Tokenize words and filter stopwords
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]

    # Remove the most common words dynamically
    common_words = set([word for word, _ in Counter(filtered_words).most_common(10)])
    filtered_words = [word for word in filtered_words if word not in common_words]

    if len(filtered_words) >= 2:
        # Apply TF-IDF weighting
        vectorizer = TfidfVectorizer(stop_words=list(stopwords))
        tfidf_matrix = vectorizer.fit_transform([" ".join(filtered_words)])
        tfidf_scores = dict(
            zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray().flatten())
        )

        # Select top N words based on TF-IDF scores
        top_words = dict(
            sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        valence_averages = calculate_valence_averages(top_words)
        word_cloud_color_palette = generate_color_palette(valence_averages)

        # Generate a StyleCloud word cloud
        stylecloud.gen_stylecloud(
            text=" ".join(top_words.keys()),
            size=3200,
            icon_name="fas fa-comment",
            colors=word_cloud_color_palette,
            background_color="black",
            output_name=output_image,
            font_path=str(PATHS.fonts / "Typo_Round_Regular_Demo.otf"),
        )

        logger.info(f"Word cloud saved as {output_image}")


#####################################################################################################################################
