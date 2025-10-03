def generate_chunk_tags(
    chunk, ai_model_results, model_name, model_key=None, chunk_key=None
):
    # Start with the original chunk tags
    updated_chunk_tags = chunk.get(chunk_key, []).copy()

    model_data = ai_model_results.get(model_name, {})
    model_results_wrapper = model_data.get("model_results", {})

    if model_key is not None:
        # model_results is a list of dicts containing 'tag'
        model_results = model_results_wrapper.get(model_key, [])

        # Extract 'tag' values from each dict, ensuring they are strings
        tags_from_model = [
            result.get("tag")
            for result in model_results
            if isinstance(result, dict) and isinstance(result.get("tag"), str)
        ]
    else:
        # model_results is expected to be a list of strings (or lists of strings)
        model_results = model_results_wrapper

        # Safely flatten if there are nested lists
        tags_from_model = []
        if isinstance(model_results, list):
            for tag in model_results:
                if isinstance(tag, str):
                    tags_from_model.append(tag)
                elif isinstance(tag, list):
                    tags_from_model.extend([t for t in tag if isinstance(t, str)])

    # Merge with the original chunk tags
    updated_chunk_tags.extend(tags_from_model)

    # Deduplicate and sort
    return sorted(set(updated_chunk_tags))
