"""Shared tokenisation and log-prob utilities for cached-reference DPO."""
import torch


def encode_answer(tok, system, user, answer, max_len):
    prompt = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    full = prompt + [{'role': 'assistant', 'content': answer}]
    p = tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                enable_thinking=False, return_dict=False)
    f = tok.apply_chat_template(full, tokenize=True, add_generation_prompt=False,
                                enable_thinking=False, return_dict=False)
    assert f[:len(p)] == p, 'chat template prefix mismatch'
    if len(p) >= len(f) or len(f) > max_len:
        return None
    labels = [-100] * len(p) + list(f[len(p):])
    return list(f), labels


def sequence_logp(logits, labels):
    """Sum log-prob of supervised tokens. logits:[B,T,V] labels:[B,T] (-100 masked)."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe = shift_labels.clamp(min=0)
    logp = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(dim=1)
