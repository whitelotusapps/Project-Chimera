#####################################################################################################################################
# NATIVE MODULES
import json
import os
from dataclasses import asdict

#####################################################################################################################################
# AI SPECIFIC MODULES
import torch

#####################################################################################################################################
# HELPER MODULES
from loguru import logger
from tqdm.auto import tqdm
from transformers import TokenClassificationPipeline
from transformers import logging as hf_logging
from transformers import pipeline
from transformers.pipelines import AggregationStrategy

from .corenlp_data import generate_corenlp_output

#####################################################################################################################################
# PROJECT SPECIFIC MODULES
from .idioms_and_beliefs import process_doc, rank_idiolect_data

#####################################################################################################################################
# Set the logging level to error to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
hf_logging.set_verbosity_error()
#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def format_score(score, decimal_places=20):
    """
    Format a floating-point score to a specified number of decimal places.

    Args:
        score (float): The numerical score to format.
        decimal_places (int, optional): The number of decimal places to format the score to. Defaults to 20.

    Returns:
        str: The formatted score as a string with the specified precision.

    Side Effects:
        None.

    Notes:
        - Excessively large `decimal_places` values may produce long strings, which could affect readability or downstream formatting expectations.
        - This function does not round beyond the limitations of floating-point precision in Python.

    Caveats:
        - Floating-point precision may introduce minor inaccuracies when converting very small or large values.
        - The output is purely string-based and is not suitable for numerical operations unless converted back to float (which may lose precision).
    """

    return f"{score:.{decimal_places}f}"


#####################################################################################################################################

#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def sequence_classifier(chunk_text, model_config_name, models):
    """
    Classify a given text chunk using a pre-loaded sequence classification model.

    Args:
        chunk_text (str): The input text to be classified.
        model_config_name (str): The key for retrieving the model, tokenizer, and pipeline from the `models` dictionary.
        db_table (str): The name of the database table associated with this classification (currently unused).
        models (dict): A dictionary containing model configurations, including model, tokenizer, and pipeline.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'label' (str): The predicted class label (human-readable if using KoalaAI/Text-Moderation).
            - 'score' (str): The classification confidence as a string with high decimal precision.

    Side Effects:
        - None directly, and logs and results may be interpreted or stored externally.
        - Uses GPU/CPU resources depending on the model's device context.

    Notes:
        - The function tokenizes and processes input text using the HuggingFace Transformers interface.
        - Applies softmax to output logits to derive probability scores for each class.
        - Supports custom label mappings for specific models like "KoalaAI/Text-Moderation".
        - Output scores are formatted as strings with up to 20 decimal places for consistency and traceability.

    Caveats:
        - Inputs exceeding 512 tokens are truncated, potentially omitting context.
        - Assumes the `models` dictionary has the required keys and that the model is loaded and compatible with Transformers API.
        - GPU memory usage can be significant for larger models.

    Important Considerations:
        - Classification accuracy and output labels are dependent on the quality and training domain of the specified model.
        - Human-readable mapping in "KoalaAI/Text-Moderation" is manually defined; updates to the model labels may require updating the mapping.
    """

    # Retrieve the model, tokenizer, and pipeline from the models dictionary
    sequence_classifier_model = models[model_config_name].get("model", None)
    sequence_classifier_tokenizer = models[model_config_name].get("tokenizer", None)
    sequence_classifier_pipeline = models[model_config_name].get("pipeline", None)

    # Identify the device of the model
    device = next(sequence_classifier_model.parameters()).device

    # Tokenize the input text and move the inputs to the same device as the model
    inputs = sequence_classifier_tokenizer(
        chunk_text,
        return_tensors="pt",
        truncation=True,  # Truncate text to the model's max length
        padding=True,  # Add padding to the text if needed
        max_length=512,  # Ensure text does not exceed the model's max length
    ).to(
        device
    )  # Move input tensors to the model's device

    # Run the model with the inputs
    outputs = sequence_classifier_model(**inputs)

    # Get the predicted logits
    logits = outputs.logits

    # Apply softmax to get probabilities (scores)
    probabilities = logits.softmax(dim=-1).squeeze().tolist()

    # Retrieve the labels
    id2label = sequence_classifier_model.config.id2label
    labels = [id2label[idx] for idx in range(len(probabilities))]

    # Combine labels and probabilities
    label_prob_pairs = list(zip(labels, probabilities))
    label_prob_pairs.sort(key=lambda item: item[1], reverse=True)

    results = []

    if model_config_name == "KoalaAI/Text-Moderation":
        # Human-readable labels based on model card
        label_definitions = {
            "S": "Sexual",
            "H": "Hate",
            "V": "Violence",
            "HR": "Harassment",
            "SH": "Self harm",
            "S3": "Sexual, minors",
            "H2": "Hate, threatening",
            "V2": "Graphic violence",
            "OK": "Okay",
        }
        # Map to human-readable labels
        for label, probability in label_prob_pairs:
            human_readable_label = label_definitions.get(label, "Unknown")
            results.append(
                {"label": human_readable_label, "score": format_score(probability)}
            )

    else:
        results = [
            {"label": label, "score": format_score(score)}
            for label, score in label_prob_pairs
        ]

    return results


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def token_classifier(chunk_text, model_config_name, models):
    """
    Perform token classification (e.g., Named Entity Recognition) on a given text chunk using a specified model.

    Args:
        chunk_text (str): The input text to classify at the token level.
        model_config_name (str): The key identifying the model configuration within the `models` dictionary.
        models (dict): A dictionary containing preloaded model configurations, including:
            - 'model': The token classification model.
            - 'tokenizer': The tokenizer associated with the model.
            - 'pipeline' (optional): Predefined pipeline (unused in this function).

    Returns:
        list[str]: A sorted list of unique entity strings identified in the input text.

    Side Effects:
        - Utilizes GPU or CPU resources depending on where the model is loaded.
        - Applies memory pressure during inference depending on the model size and token length.

    Notes:
        - Entity spans are extracted by aggregating tokens tagged with BIO-format labels (e.g., B-ORG, I-ORG).
        - Only entities of at least 4 characters and not containing '#' are returned.
        - Duplicate entities are removed and final results are sorted alphabetically.

    Caveats:
        - Only the first 512 tokens are considered due to model input constraints; longer texts will be truncated.
        - Assumes that the tokenizer and model are compatible and correctly initialized under the given `model_config_name`.
        - The 'pipeline' entry in the models dictionary is unused and may be a remnant of a previous implementation pattern.

    Important Considerations:
        - Model performance depends heavily on domain-specific training. Custom or fine-tuned models may produce more relevant entities.
        - BERT-style tokenizers may break words into subword tokens; this function reconstructs entities by joining these tokens with spaces.
        - Output labels are taken directly from the model's config (via `id2label`) and rely on proper BIO tagging conventions.
    """

    # token_classifier_model, token_classifier_tokenizer = models[model_config_name]
    token_classifier_model = models[model_config_name].get("model", None)
    token_classifier_tokenizer = models[model_config_name].get("tokenizer", None)
    token_classifier_pipeline = models[model_config_name].get("pipeline", None)

    # Identify the device of the model
    device = next(token_classifier_model.parameters()).device

    # Tokenize the input text and move the inputs to the same device as the model
    inputs = token_classifier_tokenizer(
        chunk_text,
        return_tensors="pt",
        truncation=True,  # Truncate text to the model's max length
        max_length=512,  # Ensure text does not exceed the model's max length
    ).to(
        device
    )  # Move input tensors to the model's device

    outputs = token_classifier_model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    tokens = token_classifier_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [
        token_classifier_model.config.id2label[pred.item()] for pred in predictions[0]
    ]

    entities = []
    current_entity = []
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
                current_entity = []
            current_entity.append((token, label[2:]))
        elif label.startswith("I-") and current_entity:
            current_entity.append((token, label[2:]))
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = []
    if current_entity:
        entities.append(current_entity)

    all_entities = []
    for entity in entities:
        entity_tokens = " ".join([token for token, _ in entity])
        if "#" not in entity_tokens and len(entity_tokens) >= 4:
            all_entities.append(
                {"entity_text": entity_tokens, "entity_type": entity[0][1]}
            )

    entity_texts = [entity["entity_text"] for entity in all_entities]

    # Remove duplicates and sort the list
    all_entities = sorted(set(entity_texts))

    return all_entities


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def pipeline_classifier(chunk_text, model_config_name, models, labels):
    """
    Perform zero-shot or multi-label classification on the provided text using a pipeline-based model.

    Args:
        chunk_text (str): The input text to classify.
        model_config_name (str): The key identifying the model configuration within the `models` dictionary.
        models (dict): A dictionary containing preloaded model components, including:
            - 'model': The model instance used for classification.
            - 'tokenizer': The tokenizer for preparing the input.
            - 'pipeline': The Hugging Face pipeline used to run inference.
        labels (list[str]): A list of classification labels to use for zero-shot inference.

    Returns:
        list[dict]: A list of dictionaries containing 'label' and 'score' pairs, sorted by descending score.

    Side Effects:
        - Relies on external model inference using Hugging Face pipeline, which consumes memory and compute resources.
        - Raises a ValueError if the model output format is unrecognized.

    Notes:
        - Supports both standard zero-shot classification outputs and Hugging Face zero-shot pipeline outputs with "sequence", "labels", and "scores".
        - Handles nested list output structures that may occur with some pipeline configurations.

    Caveats:
        - This function does not validate whether the labels provided match the domain or expected output of the model.
        - Model confidence scores are not calibrated and should not be interpreted as probabilities without further analysis.
        - Requires that the model and tokenizer have been correctly initialized and are compatible with the Hugging Face pipeline API.

    Important Considerations:
        - `db_table` is included in the signature but unused; ensure downstream compatibility or remove if not needed.
        - The function assumes the pipeline was created with appropriate task settings (e.g., `zero-shot-classification`).
        - Performance and accuracy depend heavily on the underlying model's training data and architecture.
    """

    pipeline_classifier_pipeline = models[model_config_name].get("pipeline", None)

    data = pipeline_classifier_pipeline(chunk_text, labels)

    # Flatten the list if it's nested
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]

    # Determine the data type and transform accordingly
    if isinstance(data, list) and all(
        isinstance(item, dict) and "label" in item and "score" in item for item in data
    ):
        # Data Type 1
        transformed_data = sorted(data, key=lambda x: x["score"], reverse=True)
    elif (
        isinstance(data, dict)
        and "sequence" in data
        and "labels" in data
        and "scores" in data
    ):
        # Data Type 2
        labels = data["labels"]
        scores = data["scores"]
        transformed_data = [
            {"label": label, "score": score} for label, score in zip(labels, scores)
        ]
        transformed_data = sorted(
            transformed_data, key=lambda x: x["score"], reverse=True
        )
    else:
        raise ValueError("Unknown data format")
    return transformed_data


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def question_answering(chunk_text, model_config_name, models, questions):
    """
    Generates question-answer (Q&A) pairs using a GLiNER-based entity classification model.

    Args:
        chunk_text (str): The input text to be queried for answers.
        model_config_name (str): Key used to select the appropriate GLiNER model from the `models` dictionary.
        models (dict): A dictionary containing preloaded models, where each entry includes a `"model"`
            capable of `predict_entities`.
        questions (list[str]): A list of questions to be used for generating Q&A pairs.

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains a "question" and a corresponding
        "answer" identified from the `chunk_text`.

    Side Effects:
        - Logs activity using `logger`, including a status message that questions are being processed.
        - Displays a progress bar via `tqdm` for visual feedback during question processing.

    Notes:
        - The model is prompted with a concatenated string of the form: `<question>\n<chunk_text>`.
        - Only questions that yield at least one match (entity labeled as "match", "answer", or "summary")
          will be included in the final result set.
        - Duplicate answers for the same question are removed using `set()` before adding to results.

    Caveats:
        - Assumes the model implements a `predict_entities` method that accepts a prompt and list of labels.
        - The label list `["match", "answer", "summary"]` is hardcoded and may need to be adapted based on
          the model's training.
        - If no matches are found for a question, it will be excluded from the results.

    Important Considerations:
        - Ensure that the model under `model_config_name` is properly loaded and supports entity extraction
          using the expected label schema.
        - The function currently does not return a record of unanswered questions.
        - The quality of Q&A generation is highly dependent on the model's training and label definitions.
    """

    logger.info("question_answering PROCESSING QUESTIONS...")
    # Extract the model from the models dictionary
    question_answering_classifier_model = models[model_config_name].get("model", None)
    question_answering_classifier_tokenizer = models[model_config_name].get(
        "tokenizer", None
    )

    # Initialize the list for questions with answers
    qna_results = []

    # Run the model to get answers for all questions
    for question_dict in tqdm(questions, desc="question"):
        # Extract the question text
        question_text = question_dict.get("question", "").strip()

        # Prepare the model input
        qna_input = {"question": question_text, "context": chunk_text}

        # Run the model
        qna_pipline = pipeline(
            "question-answering",
            model=question_answering_classifier_model,
            tokenizer=question_answering_classifier_tokenizer,
        )

        # logger.info(f"{json.dumps(qna_input, indent=4)}")
        matches = qna_pipline(qna_input)

        if matches:
            if isinstance(matches, list):
                answers = sorted(
                    set(
                        match.get("answer", "")
                        for match in matches
                        if match.get("answer")
                    )
                )
            elif isinstance(matches, dict):
                answers = [matches.get("answer", "")]
            else:
                answers = [str(matches)]

            if answers:
                result = {**question_dict, "answer": answers}
                qna_results.append(result)

    return qna_results


#####################################################################################################################################
@logger.catch
def gliner_generate_qna_results(chunk_text, model_config_name, models, questions):
    """
    Generates question-answer (Q&A) pairs using a GLiNER-based entity classification model.

    Args:
        chunk_text (str): The input text to be queried for answers.
        model_config_name (str): Key used to select the appropriate GLiNER model from the `models` dictionary.
        models (dict): A dictionary containing preloaded models, where each entry includes a `"model"`
            capable of `predict_entities`.
        questions (list[str]): A list of questions to be used for generating Q&A pairs.

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains a "question" and a corresponding
        "answer" identified from the `chunk_text`.

    Side Effects:
        - Logs activity using `logger`, including a status message that questions are being processed.
        - Displays a progress bar via `tqdm` for visual feedback during question processing.

    Notes:
        - The model is prompted with a concatenated string of the form: `<question>\n<chunk_text>`.
        - Only questions that yield at least one match (entity labeled as "match", "answer", or "summary")
          will be included in the final result set.
        - Duplicate answers for the same question are removed using `set()` before adding to results.

    Caveats:
        - Assumes the model implements a `predict_entities` method that accepts a prompt and list of labels.
        - The label list `["match", "answer", "summary"]` is hardcoded and may need to be adapted based on
          the model's training.
        - If no matches are found for a question, it will be excluded from the results.

    Important Considerations:
        - Ensure that the model under `model_config_name` is properly loaded and supports entity extraction
          using the expected label schema.
        - The function currently does not return a record of unanswered questions.
        - The quality of Q&A generation is highly dependent on the model's training and label definitions.
    """

    logger.info("PROCESSING QUESTIONS...")
    # Extract the model from the models dictionary
    gliner_classifier_model = models[model_config_name].get("model", None)

    # Initialize the list for questions with answers
    qna_results = []

    # Run the model to get answers for all questions
    for question_dict in tqdm(questions, desc="question"):
        # Extract the question text
        question_text = question_dict.get("question", "").strip()

        # Prepare the model input
        input_ = f"{question_text}\n{chunk_text}"
        model_labels = ["match", "answer", "summary"]

        # Run the model
        matches = gliner_classifier_model.predict_entities(input_, model_labels)

        if matches:
            # Collect unique answers as a sorted list
            answers = sorted(
                set(match.get("text", "") for match in matches if match.get("text"))
            )

            if answers:
                # Build a single result dictionary containing all answers
                result = {**question_dict, "answer": answers}
                qna_results.append(result)

    return qna_results

    # # Initialize the list for questions with answers
    # qna_results = []

    # # Run the model to get answers for all questions
    # for question_dict in tqdm(questions, desc="question"):
    #     # Extract the question text
    #     question_text = question_dict.get("question", "").strip()

    #     # Prepare the model input
    #     input_ = f"{question_text}\n{chunk_text}"
    #     model_labels = ["match", "answer", "summary"]

    #     # Run the model
    #     matches = gliner_classifier_model.predict_entities(input_, model_labels)

    #     if matches:
    #         # Collect unique answers
    #         answers = sorted(set([match["text"] for match in matches]))
    #         if answers:
    #             for answer in answers:
    #                 # Combine the original question data and the answer
    #                 result = {**question_dict, "answer": answer}
    #                 qna_results.append(result)

    # return qna_results

    # # Initialize the list for questions with answers and the list for Q&A pairs
    # qna_results = []

    # # Run the model to get answers for all questions
    # # for question in questions:
    # for question in tqdm(questions, desc="question"):
    #     input_ = f"{question}\n{chunk_text}"
    #     model_labels = ["match", "answer", "summary"]
    #     matches = gliner_classifier_model.predict_entities(input_, model_labels)

    #     if matches:
    #         # Collect answers for the current question
    #         answers = sorted(set([match["text"] for match in matches]))
    #         if answers:
    #             # Add each answer to the qna_list
    #             for answer in answers:
    #                 qna_results.append({"question": question, "answer": answer})

    # return qna_results


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def gliner_classifier(
    chunk_text,
    model_config_name,
    models,
    labels,
):
    """
    Classify entities in a text chunk using a GLiNER-based model and a predefined set of labels.

    Args:
        chunk_text (str): The input text to analyze for entity classification.
        model_config_name (str): Key identifying the desired model configuration in the `models` dictionary.
        models (dict): A dictionary containing loaded model components. The dictionary should contain:
            - 'model': A GLiNER-compatible model instance with a `predict_entities` method.
        labels (list[str]): A list of entity labels to guide the model's extraction process.

    Returns:
        list[dict]: A list of extracted entities, where each entity is represented as a dictionary with keys
        such as "text", "label", "start", "end", etc., depending on the model implementation.

    Side Effects:
        - None directly, though the model's internal logging or state may be updated.

    Notes:
        - This function only proceeds if the `labels` list is non-empty.
        - The `predict_entities` method of the GLiNER model is expected to accept both a string of text
          and a list of labels, and return structured prediction results.

    Caveats:
        - The function does not validate the label format or content. Invalid or out-of-domain labels may
          lead to poor results or model failure.
        - If the model under the specified `model_config_name` does not contain a valid `"model"` entry,
          the function may raise a `NoneType` error when attempting to call `predict_entities`.
        - The function currently bypasses label processing logic for empty label sets, returning nothing.

    Important Considerations:
        - Ensure that `chunk_text` is preprocessed appropriately (e.g., cleaned or segmented) before use.
        - The quality of the output is highly dependent on the specificity and relevance of the provided labels.
        - GLiNER models are typically optimized for short-to-moderate length passages and may degrade in
          accuracy or performance on long, unstructured inputs.
    """

    # Extract the model, tokenizer, and pipeline from the models dictionary
    gliner_classifier_model = models[model_config_name].get("model", None)

    # if labels and not questions:
    if len(labels) > 0:
        logger.info(f"Processing {len(labels)} labels")
        labels = gliner_classifier_model.predict_entities(chunk_text, labels)
        return labels


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def spacy_classifier(chunk_text, model_config_name, models, idiolect):
    """
    Classifies and ranks idiolect-specific linguistic features within a text chunk using a spaCy-based model.

    Args:
        chunk_text (str): The input text to be analyzed.
        model_config_name (str): Identifier for selecting the appropriate spaCy model from the `models` dictionary.
        models (dict): A dictionary of loaded models, where `models[model_config_name]` should contain a
            spaCy language model.
        idiolect (list): A list representing idiolect features (e.g., words, phrases, lexical patterns)
            to be used for determining sentences of interest.

    Returns:
        list[dict]: A list of dictionaries, each containing ranking results based on the idiolect profile.
        The structure of each dictionary is defined by the implementation of `rank_idiolect_data()`.

    Side Effects:
        - None directly. However, underlying `spaCy` components may cache pipeline runs or update model
          internal states depending on the version and configuration.

    Notes:
        - This function is designed to be modular, relying on `process_doc()` for NLP analysis and
          `rank_idiolect_data()` for scoring based on idiolect similarity or alignment.
        - The `idiolect` parameter allows customization for personalized language modeling or authorship analysis.

    Caveats:
        - The accuracy and usefulness of the results depend heavily on the quality and structure of the
          `idiolect` input and the capabilities of the loaded `spaCy` model.
        - If the selected model lacks necessary pipeline components (e.g., parser, tagger, NER),
          results may be incomplete or misleading.
        - `process_doc` and `rank_idiolect_data` must be available in the execution context and properly defined.

    Important Considerations:
        - Ensure the spaCy model is compatible with the version of spaCy installed and supports required components.
        - For best results, pre-process `chunk_text` to clean unwanted noise or encoding artifacts.
        - The function assumes that the `idiolect` object is structured appropriately to be used by the
          `rank_idiolect_data` function.
    """

    doc = process_doc(chunk_text, model_config_name, models)
    results = rank_idiolect_data(doc, model_config_name, models, idiolect)
    return results


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def keyphrase_extraction(chunk_text, model_config_name, models):
    # Initialize the pipeline using the provided model/tokenizer
    pipeline = TokenClassificationPipeline(
        model=models[model_config_name].get("model", None),
        tokenizer=models[model_config_name].get("tokenizer", None),
        aggregation_strategy=AggregationStrategy.SIMPLE,
    )

    # Run the pipeline — this already applies aggregation and postprocessing
    results = pipeline(chunk_text)

    # Extract and deduplicate keyphrases
    keyphrases = sorted({result.get("word").strip() for result in results})

    # Return as a dictionary
    return {"model_results": keyphrases}


#####################################################################################################################################


#####################################################################################################################################
@logger.catch
def generate_ai_model_results(
    chunk_text,
    model_configs,
    models,
    labels,
    questions,
    idiolect,
    chunk_calendar_start_datetime,
):
    """
    Generates AI model results by running a variety of NLP model types over the input text chunk.

    This function supports multiple model types, including `sequence_classification`, `token_classification`,
    `pipeline`, `gliner`, `spacy`, and `corenlp`, with configurable behavior for label-based classification
    and Q&A-style extraction.

    Args:
        chunk_text (str): The input text chunk to analyze.
        model_configs (list[dict]): A list of dictionaries, each specifying the configuration for a model
            (e.g., model name, type, custom flags).
        models (dict): A dictionary of loaded models, tokenizers, and pipelines keyed by model name.
        labels (list[str]): A list of label strings used for classification tasks.
        questions (list[str]): A list of question strings for Q&A-based extraction tasks.
        idiolect (list[str]): A list of idiosyncratic lexical items for use with spaCy models.
        chunk_calendar_start_datetime (datetime): The calendar-aligned start time of the text chunk, used
            primarily by CoreNLP-based analysis.

    Returns:
        dict: A dictionary containing model names as keys and dictionaries of their respective results as values.

    Side Effects:
        - Logs detailed model processing steps and configuration to the logger.
        - Prints the final results dictionary (`ai_model_results`) to standard output for inspection.

    Notes:
        - Model type handling is case-insensitive and trimmed of whitespace.
        - GLiNER-based models can perform both custom label extraction and question-answering, if enabled.
        - spaCy models are preprocessed with `process_doc()` and may reuse the same `Doc` object for multiple evaluations.
        - CoreNLP models require external server connectivity and correct annotator configuration.

    Caveats:
        - The function assumes that the models dictionary contains valid and compatible entries under each
          `model_name` key.
        - The function assumes each model type supports the corresponding method and format (e.g., `predict_entities`).
        - If no results are returned from a given model, its entry may be partially or entirely absent in the output.

    Important Considerations:
        - Duplicate processing may occur if the same model configuration is repeated without deduplication.
        - CoreNLP calls may fail silently if the external server is unavailable or misconfigured.
        - Model behavior is determined by the `model_type`, which must match expected values (e.g., "gliner", "spacy").
        - This function does not catch all runtime errors—ensure inputs are validated upstream.
    """
    dict_of_model_configs = [asdict(model) for model in model_configs]
    # logger.info(f"model_configs:\n\n{json.dumps(dict_of_model_configs, indent=4)}")
    new_chunk = True
    ai_model_results = {}

    #####################################################################################################################################
    # Start processing
    for model_config in model_configs:

        model_name = model_config.model_name
        logger.info(f"PROCESSING MODEL: {model_name}")

        #####################################################################################################################################
        # FRONT LOAD SOME MODELS PRIOR TO PROCESSING THEIR RESPECTIVE TABLES
        if model_config.model_type.strip().lower() == "spacy":
            logger.info("LOADING SPACY DOC")
            doc = process_doc(chunk_text, model_name, models)

        #####################################################################################################################################

        #####################################################################################################################################

        if model_config.model_type.strip().lower() == "keyphrase-extraction":
            logger.info("IF: Keyphrase Extraction")
            model_results = keyphrase_extraction(chunk_text, model_name, models)
            ai_model_results[model_name] = model_results
            # ai_model_results[model_name] = {
            #     "model_results": {
            #         "keyphrases": model_results,
            #     }
            # }

        if model_config.model_type.strip().lower() == "question-answering":
            logger.info("IF: question-answering")

            if len(questions) > 0:
                logger.info(f"Processing {len(questions)} questions")
                model_results = question_answering(
                    chunk_text, model_name, models, questions
                )
                if len(model_results) > 0:
                    # Initialize if needed
                    if model_name not in ai_model_results:
                        ai_model_results[model_name] = {"model_results": {}}

                    # Add qna
                    ai_model_results[model_name]["model_results"]["qna"] = model_results

        if model_config.model_type.strip().lower() == "sequence_classification":
            logger.info("IF: sequence_classification")

            model_results = sequence_classifier(chunk_text, model_name, models)
            ai_model_results[model_name] = {"model_results": model_results}

        if model_config.model_type.strip().lower() == "token_classification":
            logger.info("IF: token_classification")

            model_results = token_classifier(chunk_text, model_name, models)
            ai_model_results[model_name] = {"model_results": model_results}

        if (model_config.model_type.strip().lower() == "pipeline") or (
            model_config.model_type.strip().lower() == "gliclass"
        ):
            logger.info("IF: pipeline OR gliclass")

            model_results = pipeline_classifier(chunk_text, model_name, models, labels)
            ai_model_results[model_name] = {"model_results": model_results}

        if (
            model_config.model_type.strip().lower() == "gliner"
            and model_config.enable_custom_labels.strip().lower() == "yes"
        ):
            logger.info("IF: gliner/labels")

            model_results = gliner_classifier(
                chunk_text,
                model_name,
                models,
                labels,
            )

            # Initialize if needed
            if model_name not in ai_model_results:
                ai_model_results[model_name] = {"model_results": {}}

            # Add custom_labels
            ai_model_results[model_name]["model_results"][
                "custom_labels"
            ] = model_results

        if (
            model_config.model_type.strip().lower() == "gliner"
            and model_config.enable_qna.strip().lower() == "yes"
        ):
            logger.info("IF: gliner/Q&A")

            if len(questions) > 0:
                logger.info(f"Processing {len(questions)} questions")
                model_results = gliner_generate_qna_results(
                    chunk_text, model_name, models, questions
                )
                if len(model_results) > 0:
                    # Initialize if needed
                    if model_name not in ai_model_results:
                        ai_model_results[model_name] = {"model_results": {}}

                    # Add qna
                    ai_model_results[model_name]["model_results"]["qna"] = model_results

        if model_config.model_type.strip().lower() == "spacy":
            logger.info("IF: spacy")
            model_results = spacy_classifier(doc, model_name, models, idiolect)
            ai_model_results[model_name] = {"model_results": model_results}
        #####################################################################################################################################

        #####################################################################################################################################

        if model_config.model_type.strip().lower() == "corenlp" and new_chunk == True:
            logger.info("IF: corenlp")
            server_address = model_config.server_address
            server_port = model_config.server_port
            annotators = model_config.annotators
            pipelineLanguage = model_config.pipelineLanguage
            outputFormat = model_config.outputFormat
            ner_additional_tokensregex_rules_file = (
                model_config.ner_additional_tokensregex_rules_file
            )

            model_results = generate_corenlp_output(
                chunk_text,
                chunk_calendar_start_datetime,
                server_address,
                server_port,
                annotators,
                pipelineLanguage,
                outputFormat,
                ner_additional_tokensregex_rules_file,
            )

            ai_model_results[model_name] = {"model_results": {**model_results}}

    # print("AI_MODEL_RESULTS:\n")
    # print(f"{json.dumps(ai_model_results, indent=4)}")
    # input("")
    return ai_model_results


#####################################################################################################################################
