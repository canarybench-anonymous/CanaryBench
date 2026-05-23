"""
Loss-based evaluation for base (non-instruction) causal LMs on classification-style tasks.

For tasks like topic classification or NLI, generation is unreliable for base models
(especially DP-trained). Instead, compute the loss of each candidate label as a
continuation of the prompt and pick the lowest-loss label.
"""

from __future__ import annotations

from typing import Any

import torch


# Tasks that should use loss-based label selection (not generation).
TASK_LABELS: set[str] = {"classification", "nli"}

# Candidate labels per task (must match dataset `target` strings).
_LABELS_BY_TASK: dict[str, list[str]] = {
    "classification": ["World", "Sports", "Business", "Sci/Tech"],
    "nli": ["entailment", "contradiction", "neutral"],
}

RESPONSE_PREFIX = "\n\n### Response:\n"
PROMPT_FORMATS: set[str] = {"auto", "chat", "response"}


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def resolve_eval_prompt_format(tokenizer, prompt_format: str = "auto") -> str:
    """
    Resolve the prompt format used for evaluation.

    `auto` keeps the historical tokenizer-based behavior. Model configs can pass
    `chat` or `response` to force the same format used during finetuning.
    """
    prompt_format = (prompt_format or "auto").strip().lower()
    aliases = {
        "chat_template": "chat",
        "legacy": "response",
        "plain": "response",
        "### response": "response",
        "### response:": "response",
    }
    prompt_format = aliases.get(prompt_format, prompt_format)

    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(
            f"Unknown prompt_format={prompt_format!r}; expected one of {sorted(PROMPT_FORMATS)}"
        )

    if prompt_format == "auto":
        return "chat" if _has_chat_template(tokenizer) else "response"

    if prompt_format == "chat" and not _has_chat_template(tokenizer):
        return "response"

    return prompt_format


def build_eval_prompt(tokenizer, input_text: str, prompt_format: str = "auto") -> tuple[str, bool]:
    """
    Return the prompt string plus whether it already contains chat-template
    special tokens. Callers should tokenize chat-template prompts with
    add_special_tokens=False.
    """
    resolved_format = resolve_eval_prompt_format(tokenizer, prompt_format)
    if resolved_format == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
        ), True

    return input_text + RESPONSE_PREFIX, False


def build_eval_texts(
    tokenizer,
    input_text: str,
    label: str,
    prompt_format: str = "auto",
) -> tuple[str, str, bool]:
    """
    Return prompt-only text, full prompt+label text, and whether the text is
    chat-template formatted.
    """
    resolved_format = resolve_eval_prompt_format(tokenizer, prompt_format)
    if resolved_format == "chat":
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": label},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt, full_text, True

    prompt = input_text + RESPONSE_PREFIX
    return prompt, prompt + label, False


def _build_prompt(tokenizer, input_text: str) -> str:
    prompt, _ = build_eval_prompt(tokenizer, input_text)
    return prompt


def _avg_label_loss(
    model,
    tokenizer,
    device: str,
    prompt: str,
    full_text: str,
    *,
    max_length: int = 512,
    add_special_tokens: bool = False,
) -> float:
    """
    Compute average token loss for the label portion of `full_text`.
    Only label tokens contribute to the loss; prompt tokens are ignored.
    """
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )["input_ids"]

    full = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    input_ids = full["input_ids"].to(device)
    attention_mask = full.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = int(prompt_ids.shape[1])
    labels = input_ids.clone()
    if prompt_len > 0:
        labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(1, -1)

    mask = (shift_labels != -100).float()
    denom = mask.sum().clamp(min=1.0)
    return float((per_token * mask).sum() / denom)


def evaluate_classification_by_loss(
    model,
    tokenizer,
    dataset,
    device: str,
    *,
    max_samples: int | None = None,
    max_length: int = 512,
    prompt_format: str = "auto",
) -> list[dict[str, Any]]:
    """
    Return samples in the same schema as generation-based evaluation:
    {group, task, input, target, generated}

    Only records whose `task` is in TASK_LABELS are evaluated; others are skipped.
    """
    model.eval()
    model.to(device)

    n_total = len(dataset.records)
    if max_samples is not None and max_samples > 0:
        n_total = min(n_total, max_samples)

    samples: list[dict[str, Any]] = []

    for rec in dataset.records[:n_total]:
        task = rec.get("task", "unknown")
        if task not in TASK_LABELS:
            continue

        labels = _LABELS_BY_TASK.get(task)
        if not labels:
            continue

        best_label = None
        best_loss = None
        for label in labels:
            prompt, full_text, uses_chat_template = build_eval_texts(
                tokenizer,
                rec["input"],
                label,
                prompt_format=prompt_format,
            )
            loss = _avg_label_loss(
                model,
                tokenizer,
                device,
                prompt,
                full_text,
                max_length=max_length,
                add_special_tokens=not uses_chat_template,
            )
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_label = label

        samples.append(
            {
                "group": rec.get("group", "unknown"),
                "task": task,
                "input": rec["input"],
                "target": rec["target"],
                "generated": (best_label or "").strip(),
            }
        )

    return samples
