from collections import defaultdict

from loguru import logger

# Parameters
TOP_N_SCORED_RESULTS = 3
MIN_CHUNK_OCCURRENCE = 3
BINARY_THRESHOLD = 0.5


@logger.catch
def generate_chunk_root(chunks):
    # Initialize per-file aggregate structure
    aggregated_results = defaultdict(lambda: defaultdict(list))

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        chunk_analysis = chunk.get("chunk_analysis", {})

        for model_name, model_data in chunk_analysis.items():
            if model_name in [
                "stanford_corenlp",
                "en_core_web_sm",
                "generate_chunk_transits",
                "generate_chunk_zrs",
                "generate_chunk_profections",
                "model_name",
            ]:
                continue

            logger.info(f"model_name: {model_name}")
            model_results = model_data.get("model_results", [])

            # Case 1: flat list (e.g., keyphrases)
            if isinstance(model_results, list) and all(
                isinstance(x, str) for x in model_results
            ):
                for result in model_results:
                    aggregated_results[model_name][result].append(chunk_id)

            # Case 2: scored dicts
            elif isinstance(model_results, list) and all(
                isinstance(x, dict) for x in model_results
            ):
                total_labels = len(model_results)
                labels_with_scores = [
                    (entry.get("label"), float(entry.get("score", 0)))
                    for entry in model_results
                    if "label" in entry and "score" in entry
                ]

                for label, _ in labels_with_scores:
                    _ = aggregated_results[model_name][label]

                if total_labels >= 4:
                    for label, _ in labels_with_scores[:TOP_N_SCORED_RESULTS]:
                        aggregated_results[model_name][label].append(chunk_id)

                elif total_labels == 3:
                    label, _ = labels_with_scores[0]
                    aggregated_results[model_name][label].append(chunk_id)

                elif total_labels == 2:
                    label_1, score_1 = labels_with_scores[0]
                    label_2, score_2 = labels_with_scores[1]
                    if score_1 > BINARY_THRESHOLD:
                        aggregated_results[model_name][label_1].append(chunk_id)
                    elif score_2 > BINARY_THRESHOLD:
                        aggregated_results[model_name][label_2].append(chunk_id)
                    else:
                        aggregated_results[model_name][label_1].append(chunk_id)
            # Case 3: QnA model - special structure
            elif model_name == "knowledgator/gliner-multitask-large-v0.5":
                qna_entries = model_data.get("model_results", {}).get("qna", [])
                if not qna_entries:
                    continue

                qna_index = aggregated_results.setdefault("qna_index", {})

                for entry in qna_entries:
                    tag = entry.get("tag")
                    question = entry.get("question")
                    answers = entry.get("answer", [])

                    if not tag or not question or not answers:
                        continue

                    # Convert answer list to a single string with two newlines between items
                    formatted_answer = "\n\n".join(answers)

                    # Initialize structure for this tag
                    tag_entry = qna_index.setdefault(
                        tag,
                        {"question": question, "answers": {}, "number_of_answers": 0},
                    )

                    # Store answer by chunk_id
                    tag_entry["answers"][str(chunk_id)] = formatted_answer
                    tag_entry["number_of_answers"] = len(tag_entry["answers"])

    # Deduplicate, filter, and sort per model
    final_aggregated_results = {}

    for model_name, results in aggregated_results.items():
        if model_name == "qna_index":
            # Skip cleaning — it's already structured correctly
            final_aggregated_results[model_name] = results
            continue

        cleaned_results = {}
        is_flat_list_model = all(isinstance(x, str) for x in results.keys())

        for result, chunk_ids in results.items():
            unique_chunk_ids = sorted(set(chunk_ids))

            if is_flat_list_model:
                if len(unique_chunk_ids) >= MIN_CHUNK_OCCURRENCE:
                    cleaned_results[result] = unique_chunk_ids
            else:
                cleaned_results[result] = unique_chunk_ids

        sorted_results = dict(
            sorted(
                cleaned_results.items(),
                key=lambda item: (-len(item[1]), item[0].lower()),
            )
        )
        final_aggregated_results[model_name] = sorted_results

    return final_aggregated_results
    # Save per-file aggregate result
    # out_file_name = f"aggregated_chunk_analysis__{file.stem}.json"
    # out_file_path = output_dir / out_file_name

    # with open(out_file_path, "w", encoding="utf-8") as out_file:
    #     json.dump(final_aggregated_results, out_file, indent=4)

    # print(f"✔ Saved: {out_file_path}")
