from __future__ import annotations

import io
import math
import os
import random
from itertools import cycle
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Iterator, List, Sequence

os.environ.setdefault(
    "HF_DATASETS_CACHE",
    os.environ.get("CALIB_CACHE_DIR", "/tmp/qwen3_vl_hf_datasets_cache"),
)

import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_ID = os.environ.get("MODEL_ID", "../../Qwen/Qwen3-VL-2B-Instruct/")
QUANT_PATH = os.environ.get("QUANT_PATH", "../../Qwen/Qwen3-VL-2B-Instruct-GPTQ-Int4")
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda:0")
CALIBRATION_DEVICE = os.environ.get("CALIBRATION_DEVICE", MODEL_DEVICE)

NUM_CALIB = int(os.environ.get("NUM_CALIB", "1024"))
NUM_IMAGE_CALIB = int(
    os.environ.get(
        "NUM_IMAGE_CALIB",
        os.environ.get("NUM_COCO_CALIB", str(min(NUM_CALIB, max(1, NUM_CALIB * 3 // 4)))),
    )
)
NUM_TEXT_CALIB = int(os.environ.get("NUM_TEXT_CALIB", str(max(0, NUM_CALIB - NUM_IMAGE_CALIB))))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))

COCO_CAPTION_DATA_FILES = os.environ.get("COCO_CAPTION_DATA_FILES", "val-00001-of-00013.parquet")
COCO_CAPTION_SEED = int(os.environ.get("COCO_CAPTION_SEED", "42"))
USE_ASSISTANT_IN_IMAGE_CALIB = os.environ.get("USE_ASSISTANT_IN_IMAGE_CALIB", "0") not in {"0", "false", "False"}
REQUIRE_ASSISTANT_LANG_MATCH = os.environ.get("REQUIRE_ASSISTANT_LANG_MATCH", "1") not in {"0", "false", "False"}
IMAGE_PROMPT_MODE = os.environ.get("IMAGE_PROMPT_MODE", "mixed").lower()
USE_SYSTEM_PROMPT_IN_CALIB = os.environ.get("USE_SYSTEM_PROMPT_IN_CALIB", "1") not in {"0", "false", "False"}
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")
SYSTEM_PROMPT_ZH = os.environ.get(
    "SYSTEM_PROMPT_ZH",
    "请使用和用户问题相同的语言回答。用户用中文提问时，只用简体中文回答；用户用英文提问时，只用英文回答。不要中英混杂，不要重复同一句话。",
)
SYSTEM_PROMPT_EN = os.environ.get(
    "SYSTEM_PROMPT_EN",
    "Answer in the same language as the user's question. If the user asks in Chinese, answer only in Simplified Chinese; if the user asks in English, answer only in English. Do not mix languages or repeat the same sentence.",
)

WIKI_DATASET_ID = os.environ.get("WIKI_DATASET_ID", "wikimedia/wikipedia")
WIKI_ZH_CONFIG = os.environ.get("WIKI_ZH_CONFIG", "20231101.zh")
WIKI_EN_CONFIG = os.environ.get("WIKI_EN_CONFIG", "20231101.en")
TEXT_LANG_PATTERN = tuple(
    item.strip().lower()
    for item in os.environ.get("TEXT_LANG_PATTERN", "zh,en").split(",")
    if item.strip()
)
SHUFFLE_BUFFER = int(os.environ.get("SHUFFLE_BUFFER", "10000"))
MIN_TEXT_CHARS = int(os.environ.get("MIN_TEXT_CHARS", "128"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "1536"))

CALIB_IMAGE_MAX_PIXELS = int(os.environ.get("CALIB_IMAGE_MAX_PIXELS", str(384 * 384)))
CALIB_IMAGE_MIN_PIXELS = int(os.environ.get("CALIB_IMAGE_MIN_PIXELS", str(256 * 256)))
PRE_RESIZE_MAX_PIXELS = int(os.environ.get("PRE_RESIZE_MAX_PIXELS", str(CALIB_IMAGE_MAX_PIXELS)))
MAX_CALIB_TOKENS = int(os.environ.get("MAX_CALIB_TOKENS", "2048"))
ADD_GENERATION_PROMPT = os.environ.get("ADD_GENERATION_PROMPT", "1") not in {"0", "false", "False"}
DRY_RUN = os.environ.get("DRY_RUN", "0") in {"1", "true", "True"}

QUANT_MSE = float(os.environ.get("QUANT_MSE", "2.5"))

MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "mm_token_type_ids",
}

ZH_IMAGE_PROMPTS = [
    "请用简体中文详细描述图片内容。",
    "请用简体中文概括图片中的主要物体、场景和动作，不要重复。",
    "请观察图片，并用自然的中文回答图片里有什么。",
    "请用简体中文描述这张图片，回答要准确、简洁。",
]

EN_IMAGE_PROMPTS = [
    "Please describe this image in detail.",
    "Please identify the main objects, scene, and actions in this image.",
]


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _message_language(messages: Sequence[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return "zh" if _contains_cjk(content) else "en"
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return "zh" if _contains_cjk(str(item.get("text", ""))) else "en"
    return "en"


def _system_message(language: str) -> Dict[str, Any]:
    prompt = SYSTEM_PROMPT or (SYSTEM_PROMPT_ZH if language == "zh" else SYSTEM_PROMPT_EN)
    return {"role": "system", "content": [{"type": "text", "text": prompt}]}


def _with_optional_system(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not USE_SYSTEM_PROMPT_IN_CALIB:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages
    return [_system_message(_message_language(messages)), *messages]


def _same_language(left: str, right: str) -> bool:
    return _contains_cjk(left) == _contains_cjk(right)


def _resolve_existing_path(path_text: str) -> Path:
    input_path = Path(path_text).expanduser()
    candidates = [input_path]
    if not input_path.is_absolute():
        candidates.append(SCRIPT_DIR / input_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return input_path


def _resolve_data_files(data_files_spec: str) -> List[Path]:
    paths: List[Path] = []
    for raw_item in data_files_spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        candidate = _resolve_existing_path(item)
        glob_matches = sorted(candidate.parent.glob(candidate.name)) if any(char in candidate.name for char in "*?[]") else []
        if glob_matches:
            paths.extend(match.resolve() for match in glob_matches if match.is_file())
        elif candidate.is_file():
            paths.append(candidate.resolve())
        else:
            raise FileNotFoundError(f"Calibration parquet file not found: {item}")
    if not paths:
        raise ValueError("COCO_CAPTION_DATA_FILES must contain at least one local parquet path.")
    return paths


def _select_caption(answer: Any, rng: random.Random) -> str:
    if isinstance(answer, list):
        captions = [str(item).strip() for item in answer if str(item).strip()]
        return rng.choice(captions) if captions else ""
    return str(answer).strip()


def _caption_prompt(question: str, sample_index: int) -> str:
    clean_question = " ".join(str(question).split()).strip()
    if IMAGE_PROMPT_MODE in {"source", "original"} and clean_question:
        return clean_question
    if IMAGE_PROMPT_MODE == "mixed":
        prompts = ZH_IMAGE_PROMPTS + EN_IMAGE_PROMPTS
    elif IMAGE_PROMPT_MODE == "en":
        prompts = EN_IMAGE_PROMPTS
    else:
        prompts = ZH_IMAGE_PROMPTS
    return prompts[sample_index % len(prompts)]


def _limit_image_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    image = image.convert("RGB")
    if max_pixels <= 0 or image.width * image.height <= max_pixels:
        return image
    resize_ratio = math.sqrt(max_pixels / float(image.width * image.height))
    target_width = max(1, int(round(image.width * resize_ratio)))
    target_height = max(1, int(round(image.height * resize_ratio)))
    return image.resize((target_width, target_height), Image.Resampling.BICUBIC)


def _load_image_from_row(row: Dict[str, Any]) -> Image.Image | None:
    image_obj = row.get("image")
    if isinstance(image_obj, Image.Image):
        return _limit_image_pixels(image_obj, PRE_RESIZE_MAX_PIXELS)
    if isinstance(image_obj, dict):
        image_bytes = image_obj.get("bytes")
        if image_bytes:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return _limit_image_pixels(image, PRE_RESIZE_MAX_PIXELS)
        image_path = image_obj.get("path")
        if image_path:
            resolved_path = _resolve_existing_path(str(image_path))
            if resolved_path.exists():
                with Image.open(resolved_path) as image:
                    return _limit_image_pixels(image, PRE_RESIZE_MAX_PIXELS)
    return None


def _make_image_sample(row: Dict[str, Any], sample_index: int, rng: random.Random) -> Dict[str, Any] | None:
    caption = _select_caption(row.get("answer", ""), rng)
    question = _caption_prompt(str(row.get("question", "")), sample_index)
    if not question:
        return None
    if USE_ASSISTANT_IN_IMAGE_CALIB and not caption:
        return None

    image = _load_image_from_row(row)
    if image is None:
        return None

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    if USE_ASSISTANT_IN_IMAGE_CALIB and (not REQUIRE_ASSISTANT_LANG_MATCH or _same_language(question, caption)):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})

    return {"messages": _with_optional_system(messages), "source": "coco_caption"}


def _build_image_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    import pyarrow.parquet as pq

    rng = random.Random(COCO_CAPTION_SEED)
    data_files = _resolve_data_files(COCO_CAPTION_DATA_FILES)
    reservoir: List[Dict[str, Any]] = []
    valid_seen = 0

    for parquet_path in data_files:
        parquet_file = pq.ParquetFile(parquet_path)
        schema_names = set(parquet_file.schema_arrow.names)
        columns = [name for name in ("question", "answer", "image") if name in schema_names]
        for record_batch in parquet_file.iter_batches(batch_size=64, columns=columns):
            for row in record_batch.to_pylist():
                sample = _make_image_sample(row, valid_seen, rng)
                if sample is None:
                    continue
                valid_seen += 1
                if len(reservoir) < num_samples:
                    reservoir.append(sample)
                    if len(reservoir) % 64 == 0:
                        print(f"collected {len(reservoir)}/{num_samples} image calibration samples")
                    continue
                replacement_index = rng.randrange(valid_seen)
                if replacement_index < num_samples:
                    reservoir[replacement_index] = sample

    rng.shuffle(reservoir)
    print(f"ready image calibration samples: {len(reservoir)} from {len(data_files)} parquet file(s)")
    return reservoir


def _load_wikipedia_stream(config_name: str, seed: int):
    from datasets import load_dataset

    dataset = load_dataset(WIKI_DATASET_ID, config_name, split="train", streaming=True)
    return dataset.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)


def _normalize_text(text: str) -> str:
    compact = " ".join(text.split()).strip()
    if MAX_TEXT_CHARS > 0:
        compact = compact[:MAX_TEXT_CHARS]
    return compact


def _build_text_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    try:
        streams: Dict[str, Iterator[Dict[str, str]]] = {}
        if "zh" in TEXT_LANG_PATTERN:
            streams["zh"] = iter(_load_wikipedia_stream(WIKI_ZH_CONFIG, seed=42))
        if "en" in TEXT_LANG_PATTERN:
            streams["en"] = iter(_load_wikipedia_stream(WIKI_EN_CONFIG, seed=43))
        if not streams:
            raise ValueError("TEXT_LANG_PATTERN must contain at least one of: zh,en")
    except Exception as exc:
        print(f"warning: skip Wikipedia text calibration because dataset loading failed: {exc}")
        return []

    calibration_dataset: List[Dict[str, Any]] = []

    for language in cycle(TEXT_LANG_PATTERN):
        if len(calibration_dataset) >= num_samples:
            break
        if language not in streams:
            continue
        try:
            row = next(streams[language])
        except StopIteration:
            break
        except Exception as exc:
            print(f"warning: stop Wikipedia text calibration because streaming failed: {exc}")
            break

        text = _normalize_text(str(row.get("text", "")))
        if len(text) < MIN_TEXT_CHARS:
            continue

        calibration_dataset.append(
            {
                "messages": _with_optional_system([
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    }
                ]),
                "source": f"wikipedia_{language}",
            }
        )

        if len(calibration_dataset) % 64 == 0:
            print(f"collected {len(calibration_dataset)}/{num_samples} text calibration samples")

    print(f"ready text calibration samples: {len(calibration_dataset)}")
    return calibration_dataset


def _interleave_calibration(image_dataset: List[Dict[str, Any]], text_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not image_dataset:
        return text_dataset[:NUM_CALIB]
    if not text_dataset:
        return image_dataset[:NUM_CALIB]

    mixed_dataset: List[Dict[str, Any]] = []
    image_index = 0
    text_index = 0
    images_per_text = max(1, round(len(image_dataset) / max(1, len(text_dataset))))

    while len(mixed_dataset) < NUM_CALIB and (image_index < len(image_dataset) or text_index < len(text_dataset)):
        for _ in range(images_per_text):
            if image_index >= len(image_dataset) or len(mixed_dataset) >= NUM_CALIB:
                break
            mixed_dataset.append(image_dataset[image_index])
            image_index += 1
        if text_index < len(text_dataset) and len(mixed_dataset) < NUM_CALIB:
            mixed_dataset.append(text_dataset[text_index])
            text_index += 1

    return mixed_dataset


def build_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    requested_text_samples = min(max(NUM_TEXT_CALIB, 0), num_samples)
    requested_image_samples = min(max(NUM_IMAGE_CALIB, 0), num_samples)

    text_dataset = _build_text_calibration_dataset(requested_text_samples)
    image_target = min(num_samples - len(text_dataset), requested_image_samples + requested_text_samples - len(text_dataset))
    image_dataset = _build_image_calibration_dataset(max(0, image_target))

    calibration_dataset = _interleave_calibration(image_dataset, text_dataset)
    image_count = sum(1 for sample in calibration_dataset if sample.get("source") == "coco_caption")
    text_count = len(calibration_dataset) - image_count
    print(f"ready calibration mix: {image_count} image samples, {text_count} text samples")
    return calibration_dataset


def _has_assistant_message(messages: Sequence[Dict[str, Any]]) -> bool:
    return any(message.get("role") == "assistant" for message in messages)


def _extract_messages(example: Any) -> List[Dict[str, Any]]:
    if isinstance(example, dict):
        if "messages" in example:
            return example["messages"]
        if "role" in example and "content" in example:
            return [example]
        text = str(example.get("text", ""))
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]
    if isinstance(example, list):
        return example
    return [{"role": "user", "content": [{"type": "text", "text": str(example)}]}]


def _processor_kwargs() -> Dict[str, Any]:
    images_kwargs: Dict[str, Any] = {}
    if CALIB_IMAGE_MIN_PIXELS > 0:
        images_kwargs["min_pixels"] = CALIB_IMAGE_MIN_PIXELS
    if CALIB_IMAGE_MAX_PIXELS > 0:
        images_kwargs["max_pixels"] = CALIB_IMAGE_MAX_PIXELS
    return {"images_kwargs": images_kwargs} if images_kwargs else {}


def _get_processor(qmodel: Any):
    processor = getattr(qmodel, "processor", None)
    if processor is None and hasattr(qmodel, "load_processor"):
        processor = qmodel.load_processor()
    if processor is None:
        processor = getattr(qmodel, "tokenizer", None)
    if processor is None:
        raise RuntimeError("Qwen3-VL calibration requires a processor from GPTQModel or AutoProcessor.")
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("Qwen3-VL calibration processor must support apply_chat_template().")
    return processor


def _prepare_mixed_calibration_dataset(
    qmodel,
    calibration_dataset,
    calibration_dataset_concat_size=None,
    calibration_dataset_sort="desc",
    batch_size=1,
    calibration_data_min_length=10,
    calibration_concat_separator=None,
):
    del calibration_dataset_concat_size, calibration_concat_separator

    if batch_size != 1:
        print("multimodal calibration uses one sample per batch; ignoring batch_size > 1")

    processor = _get_processor(qmodel)
    encoded_batches: List[Dict[str, torch.Tensor]] = []
    skipped_short = 0
    skipped_long = 0
    skipped_error = 0
    processor_kwargs = _processor_kwargs()

    for sample_index, example in enumerate(calibration_dataset):
        messages = _extract_messages(example)
        try:
            encoded = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=ADD_GENERATION_PROMPT and not _has_assistant_message(messages),
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=processor_kwargs,
            )
        except Exception as exc:
            skipped_error += 1
            print(f"warning: skip calibration sample {sample_index} because processor failed: {exc}")
            continue

        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.shape[-1] <= calibration_data_min_length:
            skipped_short += 1
            continue
        if MAX_CALIB_TOKENS > 0 and input_ids.shape[-1] > MAX_CALIB_TOKENS:
            skipped_long += 1
            continue

        batch = {
            key: value.detach()
            for key, value in encoded.items()
            if key in MODEL_INPUT_KEYS and torch.is_tensor(value)
        }
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        encoded_batches.append(batch)

        if len(encoded_batches) % 64 == 0:
            image_batches = sum(1 for item in encoded_batches if "pixel_values" in item)
            print(f"encoded {len(encoded_batches)} calibration batches ({image_batches} with images)")

    if calibration_dataset_sort == "asc":
        encoded_batches.sort(key=lambda item: item["input_ids"].shape[-1])
    elif calibration_dataset_sort == "desc":
        encoded_batches.sort(key=lambda item: item["input_ids"].shape[-1], reverse=True)
    elif calibration_dataset_sort == "shuffle":
        random.Random(COCO_CAPTION_SEED).shuffle(encoded_batches)

    total_tokens = sum(batch["attention_mask"].sum().item() for batch in encoded_batches if "attention_mask" in batch)
    image_batches = sum(1 for batch in encoded_batches if "pixel_values" in batch)
    print(
        f"prepared {len(encoded_batches)} one-sample calibration batches "
        f"({image_batches} with images, {total_tokens} non-padded tokens)"
    )
    if skipped_short or skipped_long or skipped_error:
        print(f"skipped samples: short={skipped_short}, long={skipped_long}, error={skipped_error}")
    if not encoded_batches:
        raise RuntimeError("No usable calibration batches were prepared.")
    return encoded_batches


def _run_dry_prepare(calibration_dataset: List[Dict[str, Any]]) -> None:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dummy_model = SimpleNamespace(processor=processor)
    encoded_batches = _prepare_mixed_calibration_dataset(dummy_model, calibration_dataset, batch_size=1)
    first_batch = encoded_batches[0]
    print("dry run first batch:")
    for key, value in first_batch.items():
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")


def _patch_gptqmodel_sparse_position_ids() -> None:
    from gptqmodel.looper.forward_executor import ForwardExecutor

    if getattr(ForwardExecutor, "_qwen3_vl_sparse_position_ids_patched", False):
        return

    original_run_single = ForwardExecutor.run_single
    original_run_parallel = ForwardExecutor.run_parallel

    def pad_sparse_position_ids(kwargs: Dict[str, Any]) -> None:
        position_ids = kwargs.get("position_ids")
        layer_inputs = kwargs.get("layer_inputs")
        if not isinstance(position_ids, list) or layer_inputs is None:
            return
        if len(position_ids) < len(layer_inputs):
            kwargs["position_ids"] = []

    def run_single_position_ids_safe(self, *args, **kwargs):
        pad_sparse_position_ids(kwargs)
        return original_run_single(self, *args, **kwargs)

    def run_parallel_position_ids_safe(self, *args, **kwargs):
        pad_sparse_position_ids(kwargs)
        return original_run_parallel(self, *args, **kwargs)

    ForwardExecutor.run_single = run_single_position_ids_safe
    ForwardExecutor.run_parallel = run_parallel_position_ids_safe
    ForwardExecutor._qwen3_vl_sparse_position_ids_patched = True
    print("patched GPTQModel sparse position_ids replay for mixed Qwen3-VL calibration")


def main() -> None:
    print(f"model_id={MODEL_ID}")
    print(f"quant_path={QUANT_PATH}")
    print(
        "calibration="
        f"num={NUM_CALIB}, image={NUM_IMAGE_CALIB}, text={NUM_TEXT_CALIB}, "
        f"image_pixels=[{CALIB_IMAGE_MIN_PIXELS}, {CALIB_IMAGE_MAX_PIXELS}], "
        f"max_tokens={MAX_CALIB_TOKENS}"
    )

    calibration_dataset = build_calibration_dataset(NUM_CALIB)
    print(f"ready to quantize with {len(calibration_dataset)} mixed calibration samples")

    if DRY_RUN:
        _run_dry_prepare(calibration_dataset[: min(len(calibration_dataset), 8)])
        return

    from gptqmodel import GPTQModel, QuantizeConfig

    _patch_gptqmodel_sparse_position_ids()

    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        static_groups=True,
        act_group_aware=False,
        sym=True,
        mse=QUANT_MSE,
        calibration_data_device=CALIBRATION_DEVICE,
    )

    model = GPTQModel.load(MODEL_ID, quant_config, device=MODEL_DEVICE)
    model.ATTENTION_MASKS_REQUIRED_FOR_INPUT = True
    model.prepare_dataset = MethodType(_prepare_mixed_calibration_dataset, model)
    model.quantize(calibration_dataset, batch_size=BATCH_SIZE)
    model.save(QUANT_PATH)


if __name__ == "__main__":
    main()
