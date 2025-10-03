#####################################################################################################################################
# NATIVE MODULES
import gc

import spacy

#####################################################################################################################################
# AI SPECIFIC MODULES
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from gliner import GLiNER

#####################################################################################################################################
# HELPER MODULES
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

# import datetime


# import numpy as np


#####################################################################################################################################
def get_model_memory_usage(model):
    """
    Calculates the total memory usage of a PyTorch model in bytes.

    This function iterates over all parameters of the given model and computes
    the total memory usage by multiplying the number of elements by the size of
    each element in bytes.

    Args:
        model (torch.nn.Module): The PyTorch model whose memory usage is being calculated.

    Returns:
        int: The total memory usage of the model in bytes.

    Side Effects:
        None
    """

    total_memory = 0
    for param in model.parameters():
        total_memory += param.numel() * param.element_size()
    return total_memory


#####################################################################################################################################


#####################################################################################################################################
def get_gpu_memory():
    """
    Retrieves the available and total GPU memory in megabytes.

    This function checks for the availability of a CUDA-enabled GPU and, if available,
    queries memory usage statistics for device index 0. It calculates free memory as the
    difference between total memory and the sum of reserved and allocated memory. Values
    are returned in megabytes (MB).

    Returns:
        tuple[float, float]: A tuple containing:
            - free_memory_mb (float): The estimated free GPU memory in MB.
            - total_memory_mb (float): The total GPU memory in MB.
            Returns (0, 0) if no CUDA device is available.

    Side Effects:
        None

    Notes:
        - This function only queries the first GPU (`device index 0`). Systems with multiple GPUs
          will require modification to handle other devices.
        - The reported "free memory" is an approximation. CUDA memory management uses reserved
          memory blocks, so `free = total - (reserved + allocated)` may not reflect actual usable memory.
        - Requires PyTorch with CUDA support and a compatible GPU driver.
        - Will silently return (0, 0) if `torch.cuda.is_available()` is `False`; no warning or error is raised.

    Raises:
        None
    """
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - (reserved_memory + allocated_memory)
        return free_memory / (1024**2), total_memory / (1024**2)  # Return in MB
    return 0, 0


#####################################################################################################################################


#####################################################################################################################################
def load_model_to_device(model, device):
    """
    Loads a model onto the specified device (CPU or GPU) and returns it along with its memory usage.

    This function attempts to transfer a model to a given PyTorch device using the `.to()` method.
    If the model is an instance of `torch.nn.Module`, it is moved to the specified device and its
    memory usage is calculated in megabytes. Non-PyTorch models (e.g., SpaCy models) are returned
    unchanged with a memory usage value of 0.

    Args:
        model (Any): The model to be transferred. Typically a PyTorch model, but may also be a non-PyTorch
            model such as those from SpaCy.
        device (torch.device or str): The target device to load the model onto, such as `'cuda'`, `'cpu'`,
            or a specific CUDA device string (e.g., `'cuda:0'`).

    Returns:
        tuple[Any, float]: A tuple containing:
            - model: The transferred (or original) model.
            - memory_usage_mb (float): Estimated GPU memory usage in megabytes if the model is a PyTorch model.
              Returns 0 for non-PyTorch models.

    Side Effects:
        May move the model's parameters and buffers in-place to the specified device if it is a PyTorch model.

    Notes:
        - The function only attempts memory usage estimation for PyTorch models (`torch.nn.Module`).
        - Models not supporting the `.to()` method will be returned unchanged without warning.
        - The memory usage estimation is based on parameter size only and does not include temporary tensors
          or intermediate activations.
        - If the model is already on the target device, no actual memory transfer is performed.

    Raises:
        None
    """
    # Only PyTorch models support the .to() method
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        memory_usage_bytes = get_model_memory_usage(model)
        memory_usage_mb = memory_usage_bytes / (1024**2)  # Convert to MB
        return model, memory_usage_mb
    else:
        # Return the model as is for non-PyTorch models (like SpaCy models)
        return model, 0


#####################################################################################################################################


#####################################################################################################################################
def load_models(active_model_configs):
    """
    Loads and initializes all models defined in the given configuration list.

    This function supports loading multiple types of models (e.g., Hugging Face transformers,
    SpaCy models, GLiNER/GLiClass, and pipelines), either locally or from a server-based config.
    It handles memory-aware device placement (CPU vs GPU), model caching to prevent reloading,
    and graceful fallback for TensorFlow checkpoints when PyTorch weights are missing.

    Args:
        active_model_configs (list[dict]): A list of model configuration dictionaries. Each config must
            contain keys such as "model_name", "model_type", "use_model", "model_host", and
            optionally "model_pipeline_task".

    Returns:
        dict: A dictionary mapping model names to sub-dictionaries containing the following keys:
            - "model": The loaded model object (or config dict if `model_host` is "server").
            - "tokenizer": The tokenizer associated with the model, if applicable.
            - "pipeline": A callable or pipeline for inference (e.g., `transformers.pipeline`,
              GLiClass pipeline, or a custom wrapper around `model.forward()`).

    Side Effects:
        - Allocates GPU memory by loading models to device if available.
        - Caches loaded models in memory to avoid reloading from disk.
        - Prints progress bars to the console via `tqdm`.
        - Calls `gc.collect()` and `torch.cuda.empty_cache()` to manage memory.

    Notes:
        - Only models with `"use_model": "yes"` (case-insensitive) in their config are loaded.
        - Models are loaded to GPU if available, and automatically fall back to CPU if memory is insufficient.
        - Supported model types include:
            - `"sequence_classification"`
            - `"token_classification"`
            - `"zero_shot_classification"`
            - `"gliclass"` (requires custom `GLiClassModel`)
            - `"gliner"` (requires `GLiNER`)
            - `"spacy"` (requires `spacy`)
            - `"pipeline"` (uses `transformers.pipeline`)
        - Models are keyed by `model_name` in the returned dictionary.
        - If a model has already been loaded once (same `model_name`), it will not be reloaded again.
        - Server-hosted models are not loaded locally; instead, their config is passed through unchanged.

    Caveats:
        - The function assumes `torch`, `transformers`, `spacy`, and any custom model classes
          (e.g., `GLiClassModel`, `GLiNER`) are imported and available in the current environment.
        - Memory usage estimation is based on parameter size and does not account for runtime tensors.
        - Model loading exceptions are partially caught and handled only for missing PyTorch model files;
          other exceptions are re-raised.

    Raises:
        OSError: If model loading fails due to unsupported formats or missing files,
                 and cannot be resolved via fallback to TensorFlow weights.
        ValueError: If a config specifies an unsupported `model_type`.
    """

    models = {}
    cached_models = {}  # Cache loaded models to avoid reloading from disk
    gpu_free_memory, gpu_total_memory = get_gpu_memory()

    for config in tqdm(active_model_configs, leave=True):
        use_model = config.get("use_model", "")
        if use_model.strip().lower() != "yes":
            continue  # Skip this model if it's not marked for use

        model_name = config.get("model_name", "")
        model_type = config.get("model_type", "")
        model_pipeline_task = config.get("model_pipeline_task", "")
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_host = config.get("model_host", "")

        model = None
        tokenizer = None
        model_pipeline = None
        model_memory_usage = 0

        # Check if model is already cached
        if model_name in cached_models:
            model, tokenizer, model_pipeline = cached_models[model_name]
            models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "pipeline": model_pipeline,
            }
            continue

        if model_host == "local":
            try:
                if model_type == "sequence_classification":
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                elif model_type == "token_classification":
                    model = AutoModelForTokenClassification.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                elif model_type == "question-answering":
                    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                elif model_type == "keyphrase-extraction":
                    # extractor = KeyphraseExtractionPipeline(model=model_name)
                    model = AutoModelForTokenClassification.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                elif model_type == "zero_shot_classification":
                    model_pipeline = pipeline(
                        "zero-shot-classification", model=model_name
                    )
                elif model_type == "gliclass":
                    model = GLiClassModel.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model_pipeline = ZeroShotClassificationPipeline(
                        model,
                        tokenizer,
                        classification_type="multi-label",
                        progress_bar=False,
                    )
                elif model_type == "gliner":
                    model = GLiNER.from_pretrained(model_name)
                elif model_type == "spacy":
                    model = spacy.load(model_name)
                elif model_type == "pipeline":
                    model_pipeline = pipeline(model_pipeline_task, model=model_name)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

                if model_device == "cuda" and isinstance(model, torch.nn.Module):
                    model, model_memory_usage = load_model_to_device(
                        model, model_device
                    )
                    gpu_free_memory -= model_memory_usage

                if gpu_free_memory < 0 and isinstance(model, torch.nn.Module):
                    model_device = "cpu"
                    model, model_memory_usage = load_model_to_device(
                        model, model_device
                    )

            except OSError as e:
                if "does not appear to have a file named pytorch_model.bin" in str(e):
                    if model_type == "sequence_classification":
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_name, from_tf=True
                        )
                    elif model_type == "token_classification":
                        model = AutoModelForTokenClassification.from_pretrained(
                            model_name, from_tf=True
                        )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)

                    if model_device == "cuda":
                        model, model_memory_usage = load_model_to_device(
                            model, model_device
                        )
                        gpu_free_memory -= model_memory_usage

                    if gpu_free_memory < 0:
                        model_device = "cpu"
                        model, model_memory_usage = load_model_to_device(
                            model, model_device
                        )
                else:
                    raise e

            if isinstance(model, torch.nn.Module):

                def model_predict(inputs):
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    return outputs

                model_pipeline = model_predict

            cached_models[model_name] = (model, tokenizer, model_pipeline)

            models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "pipeline": model_pipeline,
            }

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elif model_host == "server":
            models[model_name] = {**config}

    return models


#####################################################################################################################################
