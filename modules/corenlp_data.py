#####################################################################################################################################
# NATIVE MODULES
import requests

#####################################################################################################################################
# HELPER MODULES
from loguru import logger


#####################################################################################################################################
def corenlp_annotate_text(
    chunk,
    chunk_date,
    corenlp_server_url,
    annotators,
    pipelineLanguage,
    outputFormat,
    ner_additional_tokensregex_rules,
):
    """
    Sends a text chunk to a Stanford CoreNLP server for annotation and returns the processed sentence data.

    This function configures and executes a POST request to a CoreNLP server using specified annotators
    and other pipeline properties. The server response is expected in JSON format and parsed to extract
    sentence-level annotations.

    Args:
        chunk (str): The text content to be annotated.
        chunk_date (str): A string representing the date context (e.g., for temporal analysis), passed as the "date" property.
        corenlp_server_url (str): The URL endpoint of the running CoreNLP server.
        annotators (str): Comma-separated string of annotators to apply (e.g., "tokenize,ssplit,pos,ner").
        pipelineLanguage (str): The language of the pipeline (e.g., "en" for English).
        outputFormat (str): The desired output format (must be "json" for this function to work correctly).
        ner_additional_tokensregex_rules (str): Filename of the additional NER TokensRegex rules to apply.

    Returns:
        list: A list of annotated sentence dictionaries from the CoreNLP server response.

    Side Effects:
        - Logs information about the annotation process.
        - Performs an external HTTP POST request to the CoreNLP server.

    Notes:
        - Requires a running and accessible Stanford CoreNLP server instance with the necessary annotators loaded.
        - The `ner.additional.tokensregex.rules` path must be relative to the server's working directory or correctly mapped.

    Caveats:
        - This function assumes the server returns a JSON response containing a "sentences" field.
        - If the server is unreachable or misconfigured, this function will raise a runtime exception from `requests`.
        - If `outputFormat` is not set to "json", `response.json()` will fail.
        - TokensRegex rule file must be correctly formatted and compatible with the CoreNLP pipeline version in use.
    """

    logger.info("GENERATING COREBNLP DATA")
    properties = {
        "annotators": annotators,
        "pipelineLanguage": pipelineLanguage,
        "outputFormat": outputFormat,
        "date": f"{chunk_date}",
        "ner.additional.tokensregex.rules": f"./{ner_additional_tokensregex_rules}",
        #'regexner.verbose': 'true' FOR TROUBLE SHOOTING
    }

    response = requests.post(
        corenlp_server_url,
        params={"properties": str(properties)},
        data=chunk.encode("utf-8"),
    )
    response_json = response.json()
    sentences = response_json["sentences"]

    return sentences


#####################################################################################################################################


#####################################################################################################################################
def is_not_empty(value):
    """
    Checks whether a given value is considered non-empty.

    This function determines if a value is neither `None`, `0`, nor an empty iterable
    (i.e., empty `str`, `list`, `dict`, or `set`).

    Args:
        value (Any): The value to check.

    Returns:
        bool: `True` if the value is not `None`, not `0`, and not an empty iterable; `False` otherwise.

    Side Effects:
        None.

    Notes:
        - This function does not consider other falsy values such as `False`, `''` (empty string),
          or empty tuples unless they are specifically one of the supported types.
        - If other data types (e.g., custom objects, numbers other than 0) are passed,
          they may return `True` depending on their truthiness and support for `len()`.

    Caveats:
        - `0` is always considered empty regardless of type, which may not be appropriate
          for all use cases (e.g., distinguishing between `0` and `None`).
        - Does not handle all iterable types (e.g., `tuple` is not checked explicitly).
    """

    # Check if the value is None, 0, or an empty iterable
    if value is None or value == 0:
        return False
    if isinstance(value, (str, list, dict, set)) and not value:
        return False
    return True


#####################################################################################################################################


#####################################################################################################################################
def corenlp_sentiment(chunk_sentences):
    """
    Calculates the average sentiment distribution across a list of sentence-level sentiment data.

    This function aggregates sentiment distributions for each sentence, then computes
    the average sentiment distribution across all provided sentences. Each sentence is
    expected to contain a `sentimentDistribution` key with a list of probabilities
    corresponding to different sentiment categories.

    Args:
        chunk_sentences (list): A list of dictionaries, where each dictionary represents a sentence
            annotated with a "sentimentDistribution" key containing a list of sentiment probabilities.

    Returns:
        list: A list of floats representing the average sentiment probabilities across all sentences,
            one value for each sentiment category (e.g., very negative to very positive in 5-category models).

    Side Effects:
        Logs a message using the global `logger` to indicate sentiment processing has started.

    Notes:
        - This function assumes that all `sentimentDistribution` lists are of equal length.
        - CoreNLP sentiment typically includes five categories: very negative, negative, neutral, positive,
          and very positive. Adjust interpretation accordingly based on the model used.

    Caveats:
        - No validation is performed to ensure the distributions sum to 1 or contain valid probabilities.
        - If `chunk_sentences` is empty or improperly formatted, the function may raise an exception.
        - Only the first sentence is used to determine the number of sentiment categories, which may lead
          to incorrect behavior if inconsistent inputs are provided.
    """

    logger.info("CoreNLP Sentiment")
    sentiment_distributions = []
    total_number_of_sentiments = len(chunk_sentences)

    for chunk_sentence in chunk_sentences:
        sentiment_distributions.append(chunk_sentence["sentimentDistribution"])

    # Initialize an array to store the cumulative probabilities
    num_categories = len(
        sentiment_distributions[0]
    )  # Determine the number of categories from the first distribution
    cumulative_sentiments = [0.0] * num_categories

    # Sum the sentimentDistribution values for each category
    for distribution in sentiment_distributions:
        for i in range(len(distribution)):
            cumulative_sentiments[i] += distribution[i]

    # Calculate the average sentiment for each category
    average_sentiments = [x / total_number_of_sentiments for x in cumulative_sentiments]

    return average_sentiments


#####################################################################################################################################


#####################################################################################################################################
def corenlp_sentence_enetity_mentions(chunk_sentences):
    """
    Extracts and returns unique named entity mentions from CoreNLP-annotated sentences.

    This function processes a list of sentences annotated by CoreNLP, extracting
    entity mentions that are not labeled as "O" (outside named entities) and filtering
    out common pronouns. It returns unique named entities with details, their names,
    and their types, each sorted alphabetically.

    Args:
        chunk_sentences (list): A list of dictionaries representing sentences annotated by CoreNLP,
            each containing an "entitymentions" key with a list of entity mention dictionaries.

    Returns:
        tuple:
            - sorted_unique_all_chunk_NER_details (list of dict): Unique named entity mentions with keys
              "text", "ner", and optionally "normalizedNER", sorted alphabetically by text.
            - sorted_unique_all_chunk_NER_names (list of str): Sorted unique entity mention texts.
            - sorted_unique_all_chunk_NER_types (list of str): Sorted unique entity types (NER labels).

    Side Effects:
        Logs a message indicating that CoreNLP NER processing has started.

    Notes:
        - Entities with labels "O" or those matching a predefined list of pronouns
          ("he", "she", "they", etc.) are excluded.
        - Deduplication preserves the first occurrence of each entity text.
        - The function expects the input to have consistent CoreNLP output format.

    Caveats:
        - If "entitymentions" is missing or not a list in any sentence, this may raise an exception.
        - Normalized NER field may be missing for some entity mentions, defaulting to empty string.
        - Sorting is case-sensitive and done lexicographically.
    """

    logger.info("CoreNLP NER")
    all_chunk_NER_details = []
    all_chunk_NER_names = []
    all_chunk_NER_types = []

    ners_to_remove = ["he", "she", "they", "them", "him", "her", "his", "hers"]
    for chunk_sentence in chunk_sentences:
        if is_not_empty(chunk_sentence["entitymentions"]):
            for entity_mention in chunk_sentence["entitymentions"]:
                if (
                    entity_mention["ner"] != "O"
                    and str(entity_mention["text"]).lower() not in ners_to_remove
                ):
                    entity_mention_data = {
                        "text": entity_mention["text"],
                        "ner": entity_mention["ner"],
                        "normalizedNER": entity_mention.get("normalizedNER", ""),
                    }
                    all_chunk_NER_details.append(entity_mention_data)
                    all_chunk_NER_names.append(entity_mention["text"])
                    all_chunk_NER_types.append(entity_mention["ner"])

    # Remove duplicates from all_mentions while preserving order
    unique_mentions = {}
    for single_NER in all_chunk_NER_details:
        text = single_NER["text"]
        if text not in unique_mentions:
            unique_mentions[text] = single_NER

    # Extract the values from the dictionary and sort them by text
    sorted_unique_all_chunk_NER_details = sorted(
        unique_mentions.values(), key=lambda x: x["text"]
    )

    # Sort and remove duplicates from text_mentions and ner_mentions
    sorted_unique_all_chunk_NER_names = sorted(set(all_chunk_NER_names))
    sorted_unique_all_chunk_NER_types = sorted(set(all_chunk_NER_types))

    return (
        sorted_unique_all_chunk_NER_details,
        sorted_unique_all_chunk_NER_names,
        sorted_unique_all_chunk_NER_types,
    )


#####################################################################################################################################


#####################################################################################################################################
def corenlp_sorted_sentiment_meaning(average_sentiments):
    """
    Maps average sentiment scores from CoreNLP to sentiment categories, sorts them by score,
    and returns a list of sentiments with their corresponding scores in descending order.

    Args:
        average_sentiments (list of float): A list of five average sentiment scores corresponding
            to CoreNLP sentiment categories in the order:
            [Very Negative, Negative, Neutral, Positive, Very Positive].

    Returns:
        list of dict: A list of dictionaries each containing:
            - "sentiment" (str): Sentiment category name ("very_negative", "negative", "neutral",
              "positive", "very_positive").
            - "score" (float): The average score for the sentiment category.
        The list is sorted by score in descending order.

    Side Effects:
        Logs an informational message indicating the processing of CoreNLP sentiment sorting.

    Notes:
        - The sentiment categories correspond to the standard CoreNLP sentimentDistribution indices.
        - Scores should be normalized probabilities or average values summing approximately to 1.
        - Sorting allows identification of the dominant sentiment(s) in the analyzed text.

    Caveats:
        - Input list must have exactly five elements; otherwise, index errors may occur.
        - The function assumes the input order strictly follows CoreNLP sentiment categories.
    """

    logger.info("CoreNLP Sorted Sentiment")
    """
    In Stanford CoreNLP, the sentimentDistribution array represents the distribution of sentiment scores for a given sentence or text segment, where each element in the array corresponds to a specific sentiment category. The categories are usually represented as follows:

    0: Very Negative
    1: Negative
    2: Neutral
    3: Positive
    4: Very Positive
    """

    sentiment_meaning = {
        "very_negative": average_sentiments[0],
        "negative": average_sentiments[1],
        "neutral": average_sentiments[2],
        "positive": average_sentiments[3],
        "very_positive": average_sentiments[4],
    }

    # Sort the dictionary by values in descending order
    sorted_sentiment_meaning = sorted(
        sentiment_meaning.items(), key=lambda item: item[1], reverse=True
    )

    # Convert to a list of dictionaries with "sentiment" and "score" keys
    sentiment_list = [
        {"sentiment": sentiment, "score": score}
        for sentiment, score in sorted_sentiment_meaning
    ]

    return sentiment_list


#####################################################################################################################################


#####################################################################################################################################
def generate_corenlp_output(
    chunk,
    chunk_date,
    corenlp_server_address,
    corenlp_server_port,
    annotators,
    pipelineLanguage,
    outputFormat,
    ner_additional_tokensregex_rules,
):
    """
    Generates structured CoreNLP output for a given text chunk by performing annotation,
    sentiment analysis, and named entity recognition (NER).

    This function communicates with a running Stanford CoreNLP server to annotate the provided
    text, compute sentiment distributions, extract and sort average sentiment values, and
    identify named entity mentions with associated metadata.

    Args:
        chunk (str): The text to be annotated and analyzed.
        chunk_date (str): A string representing the date context for annotation (e.g., used in temporal processing).
        corenlp_server_address (str): Base URL of the CoreNLP server (e.g., "http://localhost").
        corenlp_server_port (str or int): Port number on which the CoreNLP server is running.
        annotators (str): Comma-separated string specifying the CoreNLP annotators to apply (e.g., "tokenize,ssplit,pos").
        pipelineLanguage (str): The language code (e.g., "en") for the annotation pipeline.
        outputFormat (str): Desired format of CoreNLP output, typically "json".
        ner_additional_tokensregex_rules (str): Filename of additional NER rules to include in the annotation.

    Returns:
        dict: A dictionary containing annotated CoreNLP results with the following keys:
            - "sentiment" (list of dict): Sorted average sentiment categories and scores.
            - "ner_details" (list of dict): Unique NER entity details with text, type, and normalized form.
            - "ner_names" (list of str): Sorted list of unique entity mention texts.
            - "ner_types" (list of str): Sorted list of unique NER types.
            - "sentences" (list): Raw sentence-level annotation output from CoreNLP.

    Side Effects:
        - Sends HTTP POST requests to the CoreNLP server.
        - Logs informational messages during each major processing step.

    Notes:
        - The NER filter excludes common pronouns and non-entity labels.
        - Sentiment is calculated across all sentences, and the results are averaged and sorted.
        - All data returned is based on the response from the external CoreNLP server.

    Caveats:
        - Requires the CoreNLP server to be running and accessible at the given address/port.
        - Network failures or malformed CoreNLP configurations may result in runtime exceptions.
        - The function assumes that CoreNLP returns expected keys such as "sentences" and "entitymentions".
    """

    corenlp_server_url = f"{corenlp_server_address}:{corenlp_server_port}"

    chunk_sentences = corenlp_annotate_text(
        chunk,
        chunk_date,
        corenlp_server_url,
        annotators,
        pipelineLanguage,
        outputFormat,
        ner_additional_tokensregex_rules,
    )

    chunk_sentiment = corenlp_sorted_sentiment_meaning(
        corenlp_sentiment(chunk_sentences)
    )
    chunk_NER_details, chunk_NER_names, chunk_NER_types = (
        corenlp_sentence_enetity_mentions(chunk_sentences)
    )
    # chunk_NER_details = corenlp_sentence_enetity_mentions(chunk_sentences)

    corenlp_dict = {
        "sentiment": chunk_sentiment,
        "ner_details": chunk_NER_details,
        "ner_names": chunk_NER_names,
        "ner_types": chunk_NER_types,
        "sentences": chunk_sentences,
    }

    logger.info("Returning CoreNLP Results")
    return corenlp_dict
