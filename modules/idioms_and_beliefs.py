#####################################################################################################################################
# NATIVE MODULES
import re

idiolect = None


#####################################################################################################################################
# Function to expand contractions
#####################################################################################################################################
def expand_contractions(text, contractions_dict):
    """
    Expands contractions in a given text using a provided dictionary of contraction mappings.

    This function identifies and replaces contracted words (e.g., "don't", "can't") with their expanded
    forms (e.g., "do not", "cannot") based on the given `contractions_dict`.

    Args:
        text (str): The input text potentially containing contractions.
        contractions_dict (dict): A dictionary where keys are contracted forms and values are their expansions.

    Returns:
        str: The text with all matched contractions expanded.

    Side Effects:
        None.

    Notes:
        - The contraction matching is case-sensitive. Ensure that keys in `contractions_dict` match the casing
          used in the input text.
        - Regular expressions are compiled dynamically from the dictionary keys at each call, which may affect
          performance for very large dictionaries or repeated use.

    Caveats:
        - No normalization is performed before matching. For example, "Don't" (with curly quote or capital 'D')
          will not match "don't" unless explicitly included in the dictionary.
        - If the contraction appears in a different form (e.g., spaced or with extra punctuation), it may not match.

    Important Considerations:
        - Ensure the contractions dictionary does not contain overlapping or ambiguous entries, as regex alternations
          are evaluated in order and may produce unexpected substitutions.
        - For performance-critical use, consider precompiling and caching the regex pattern outside of this function.
    """

    # contractions_re = re.compile("(%s)" % "|".join(contractions_dict.keys()))

    # def replace(match):
    #     return contractions_dict[match.group(0)]

    # return contractions_re.sub(replace, text)

    contractions_re = re.compile("(%s)" % "|".join(re.escape(k) for k in contractions_dict.keys()))

    def replace(match):
        return contractions_dict[match.group(0)]

    lines = text.splitlines()
    all_lines = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        all_lines.add(line)
        expanded_line = contractions_re.sub(replace, line)
        all_lines.add(expanded_line)

    return "\n".join(sorted(all_lines))


#####################################################################################################################################


#####################################################################################################################################
def load_iodlect(idiolect_file_path, contractions_dict):
    """
    Loads and processes a list of idiomatic phrases (idiolect) from a file, expanding contractions
    and ensuring uniqueness of entries.

    This function reads idioms from a text file, strips whitespace, expands any contractions using
    the provided dictionary, and stores both original and expanded forms in a deduplicated, sorted list.
    The resulting list is stored in a global variable `idiolect`.

    Args:
        idiolect_file_path (str): Path to the file containing idiomatic phrases, one per line.
        contractions_dict (dict): A dictionary mapping contractions (e.g., "can't") to their expanded forms (e.g., "cannot").

    Returns:
        tuple:
            int: The number of original idiomatic lines loaded from the file.
            list[str]: A sorted list containing both the original and expanded idioms, with duplicates removed.

    Side Effects:
        - Modifies the global variable `idiolect` by assigning it the deduplicated and sorted list of phrases.
        - Opens and reads from a file on disk.

    Notes:
        - The idioms are stored with both their original and contraction-expanded forms to improve downstream
          matching and classification coverage.
        - The final list contains unique entries only, combining original and expanded phrases.

    Caveats:
        - The function only processes the file once per runtime session, as it checks if `idiolect` is already loaded.
        - If the `contractions_dict` is incomplete or inaccurate, expanded results may be misleading.
        - File must be UTF-8 encoded; otherwise, decoding errors may occur.

    Important Considerations:
        - Ensure that `contractions_dict` covers all relevant forms found in the idiom file.
        - If this function is called again after `idiolect` is populated, it will return the previously loaded result
          and not reprocess the file or contractions.
    """

    global idiolect
    if idiolect is None:
        with open(idiolect_file_path, "r", encoding="utf-8") as file:
            idiolect = [line.strip() for line in file.readlines()]

        idiolect = sorted(set(idiolect))

        # Create sets to store unique original and expanded lines
        unique_original_idioms = set()
        unique_expanded_idioms = set()

        for idiom in idiolect:
            original = idiom.strip()
            expanded = expand_contractions(original, contractions_dict)
            unique_original_idioms.add(original)
            unique_expanded_idioms.add(expanded)

        # Combine original and expanded lines into one set
        expanded_idiolect = sorted(unique_original_idioms | unique_expanded_idioms)
        idiolect_count = len(idiolect)
    return idiolect_count, expanded_idiolect


#####################################################################################################################################


#####################################################################################################################################
def unload_idiolect():
    """
    Unloads the global `idiolect` variable by resetting it to `None`.

    This function clears the in-memory representation of the idiolect list,
    allowing it to be reloaded or reprocessed later.

    Args:
        None

    Returns:
        None

    Side Effects:
        - Modifies the global variable `idiolect`, setting it to `None`.

    Notes:
        - This function is useful for memory management or reinitializing the idiolect data
          in long-running applications or dynamic workflows.

    Caveats:
        - After unloading, any function that relies on `idiolect` should reload it using
          `load_idiolect()` before proceeding.
        - If used unintentionally, this may result in runtime errors where `idiolect` is expected
          to be a list.

    Important Considerations:
        - Use this function when a clean state is needed or when updating the idiolect source file.
    """

    global idiolect
    idiolect = None


#####################################################################################################################################


#####################################################################################################################################
# SPACY VERSION
# BELOW IS GOLDEN CODE!!!!
#####################################################################################################################################
def get_causes(chunk_sentence, spacy_classifier_model):
    """
    Extracts cause-effect relationships from a sentence using a spaCy dependency parser.

    This function analyzes the syntactic structure of a sentence and attempts to identify
    causal agents (subjects), actions (verbs), affected objects, and related entities
    based on prepositional phrases.

    Args:
        chunk_sentence (str): The sentence to analyze for causal relationships.
        spacy_classifier_model (Callable): A loaded spaCy language model used for parsing
            the sentence.

    Returns:
        dict: A dictionary where each key is an action verb (str), and the value is another
        dictionary containing:
            - 'subject' (str): The identified subject of the action.
            - 'object' (str): The direct or indirect object affected by the action.
            - 'entities' (dict): A mapping of prepositions to lists of related noun phrases
              or pronouns.

    Side Effects:
        None

    Notes:
        - Handles possessive structures (e.g., "his failure") and attempts to recover
          entity references from pronouns.
        - Recognizes objects via multiple syntactic roles: `dobj`, `attr`, `prep > pobj`,
          and `obl` structures.
        - Aggregates related entities from prepositional phrases and conjunct nouns.

    Caveats:
        - Only one causal action per subject is returned due to the `break` statement after
          a successful object match.
        - May misinterpret or miss relationships in grammatically complex or ambiguous sentences.
        - Assumes English syntax and a compatible spaCy language model.

    Important Considerations:
        - Performance and accuracy are highly dependent on the quality and configuration of
          the provided `spacy_classifier_model`.
        - Designed for simple declarative sentences; may not capture causes embedded within
          clauses, questions, or ellipses.
    """

    doc = spacy_classifier_model(chunk_sentence)
    causes = {}
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token
            poss_token = None
            for child in token.children:
                if child.dep_ == "poss":
                    poss_token = child
                    break
            if subject.pos_ == "NOUN" and subject.nbor().dep_ == "cop":
                subject = subject.nbor()
            for ancestor in token.ancestors:
                if ancestor.pos_ == "VERB":
                    action = ancestor
                    obj = None
                    for child in action.children:
                        if child.dep_ == "dobj" or child.dep_ == "attr":
                            obj = child
                            break
                        elif child.dep_ == "prep":
                            for subchild in child.children:
                                if subchild.dep_ == "pobj":
                                    obj = subchild
                                    break
                            if obj is None and child.text == "of":
                                for subchild in child.children:
                                    if subchild.dep_ == "pobj":
                                        obj = subchild
                                        break
                        elif child.dep_ == "obl":
                            for subchild in child.children:
                                if subchild.text == "of":
                                    obj = subchild.children.__next__()
                                    break
                    if obj is not None:
                        entities = {}
                        for child in action.children:
                            if child.dep_ == "prep":
                                prep = child.text
                                entity_tokens = []
                                for subchild in child.children:
                                    if subchild.dep_ == "pobj":
                                        entity_tokens.append(subchild)
                                    elif subchild.dep_ == "compound":
                                        entity_tokens.append(subchild)
                                    elif (
                                        subchild.dep_ == "conj"
                                        and subchild.pos_ == "NOUN"
                                    ):
                                        entity_tokens.append(subchild)
                                for entity_token in entity_tokens:
                                    entity_text = entity_token.text
                                    if entity_token.pos_ == "PRON":
                                        for ancestor in entity_token.ancestors:
                                            if ancestor.pos_ == "NOUN":
                                                entity_text = ancestor.text
                                                break
                                    if entity_token.n_rights > 0:
                                        entity_text = " ".join(
                                            [entity_text]
                                            + [tok.text for tok in entity_token.rights]
                                        )
                                    if prep not in entities:
                                        entities[prep] = []
                                    entities[prep].append(entity_text)
                        if poss_token:
                            subject_text = f"{poss_token.text} {subject.text}"
                        else:
                            subject_text = subject.text
                        causes[action.text] = {
                            "subject": subject_text,
                            "object": obj.text,
                            "entities": entities,
                        }
                        break

    return causes


#####################################################################################################################################


#####################################################################################################################################
def process_doc(chunk_text, model_config_name, models):
    """
    Processes a text chunk using a spaCy model to generate a parsed `Doc` object.

    This function retrieves the specified spaCy model from the `models` dictionary
    using the given configuration name and applies it to the input text to obtain
    tokenized, parsed, and annotated linguistic data.

    Args:
        chunk_text (str): The text to be analyzed by the spaCy model.
        model_config_name (str): The key used to retrieve the correct model configuration
            from the `models` dictionary.
        models (dict): A dictionary containing loaded model configurations, each with keys
            like "model", "tokenizer", and "pipeline".

    Returns:
        spacy.tokens.Doc: A spaCy `Doc` object containing tokenized and linguistically
        annotated information about the input text.

    Side Effects:
        None

    Notes:
        - Only the "model" key is used from the specified `model_config_name` configuration.
          The "tokenizer" and "pipeline" keys are currently unused, but may be required in
          future extensions.
        - The returned `Doc` object can be further used for named entity recognition (NER),
          dependency parsing, part-of-speech tagging, etc.

    Caveats:
        - If the model is not found under the given `model_config_name`, `None` is returned
          from `models[model_config_name].get("model", None)`, which will raise a
          `TypeError` when called.
        - Assumes that the spaCy model is already loaded and correctly registered in the
          `models` dictionary.
        - If the model is not a callable spaCy object, runtime failure will occur.
    """

    spacy_classifier_model = models[model_config_name].get("model", None)

    doc = spacy_classifier_model(chunk_text)
    return doc


#####################################################################################################################################
def rank_idiolect_data(chunk_text, model_config_name, models, idiolect):
    """
    Analyzes a text chunk to identify and rank idiom matches and actions related to personal expression.

    This function uses a spaCy model to parse the given text and identify idioms found in the
    provided idiolect. It also attempts to extract actions (verbs) directed at "me" to highlight
    personalized narrative elements. Results are ranked and returned in structured format.

    Args:
        chunk_text (str): The full text to be analyzed.
        model_config_name (str): The key identifying the spaCy model in the `models` dictionary.
        models (dict): A dictionary containing pre-loaded model components such as spaCy model, tokenizer, and pipeline.
        idiolect (list of str): A list of idioms or key phrases representing a unique personal or linguistic fingerprint.

    Returns:
        dict: A dictionary containing:
            - "sentences": Ranked list of sentences with significant idiom matches.
            - "actions": List of detected actions related to the subject "me", including context and grammatical causes.

    Side Effects:
        - Calls the `get_causes()` function which performs syntactic dependency parsing to extract subject-verb-object relationships.

    Notes:
        - Sentences with at least 4 unique idioms from the idiolect are stored and ranked based on the number and frequency of matches.
        - Context includes up to 5 sentences prior to the sentence of interest.
        - Verbs associated with the pronoun "me" are captured for psychological or narrative interpretation.
        - Idiom matching is case-insensitive and uses regular expressions for boundary-safe detection.

    Caveats:
        - The function assumes the spaCy model associated with `model_config_name` is already loaded and callable.
        - Matching logic does not support fuzzy matching; only exact word-boundary matches are considered.
        - If idiom matches are detected and the sentence fails to meet the threshold, it may be skipped entirely.
        - Assumes English language tokenization and POS tagging conventions.
    """

    spacy_classifier_model = models[model_config_name].get("model", None)

    ranked_idiolect_data = {"actions": [], "sentences": []}
    doc = spacy_classifier_model(chunk_text)
    for i, sentence in enumerate(doc.sents):
        causes = get_causes(sentence.text, spacy_classifier_model)
        matched_idioms = {}
        all_matches = []

        # Check for exact single-word and multi-word idiom matches in the idiolect
        sent_text = sentence.text.lower()
        for idiom in idiolect:
            pattern = r"\b" + re.escape(idiom.lower()) + r"\b"
            matches = re.findall(pattern, sent_text)
            all_matches.extend(matches)

        # Count the occurrences of the longest matching phrases
        for match in set(all_matches):
            longest_idioms = [idiom for idiom in set(all_matches) if match in idiom]
            longest_idiom = max(longest_idioms, key=len)
            count = all_matches.count(longest_idiom)
            matched_idioms[longest_idiom] = matched_idioms.get(longest_idiom, 0) + count

        if matched_idioms and len(matched_idioms.keys()) >= 4:
            context_sentences = list(doc.sents)[max(0, i - 5) : i]
            context = [sentence.text for sentence in context_sentences]
            sorted_idioms = dict(
                sorted(matched_idioms.items(), key=lambda item: -item[1])
            )
            idiom_matches = {
                "idioms": sorted_idioms,
                "number_of_matches": len(sorted_idioms.keys()),
                "sum_of_matches": sum(sorted_idioms.values()),
            }
            ranked_idiolect_data["sentences"].append(
                {
                    "context": context,
                    "sentence_of_interest": sentence.text,
                    "idiom_matches": idiom_matches,
                }
            )
        elif (
            ranked_idiolect_data["sentences"]
            and "idiom_matches" not in ranked_idiolect_data["sentences"][-1]
        ):
            ranked_idiolect_data["sentences"].pop()
        for token in sentence:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.text == "me":
                        action = token.text
                        context_sents = list(doc.sents)[max(0, i - 5) : i]
                        context = [sent.text for sent in context_sents]
                        ranked_idiolect_data["actions"].append(
                            {
                                "action": action,
                                "object": child.text,
                                "context": context,
                                "action_sentence": sentence.text,
                                "actions": causes,
                            }
                        )
                        break
    ranked_idiolect_data["sentences"] = sorted(
        ranked_idiolect_data["sentences"],
        key=lambda x: -x["idiom_matches"]["sum_of_matches"],
    )
    return ranked_idiolect_data


#####################################################################################################################################
