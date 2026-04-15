"""Model-family color mapping and canonical label helpers."""
from __future__ import annotations


def _build_model_color_map(models: list[str]) -> dict[str, tuple[float, float, float, float]]:
    import matplotlib.cm as cm

    family_bases = {
        "deepseek": cm.Blues,
        "gemini": cm.Greens,
        "gpt": cm.Reds,
    }
    grouped: dict[str, list[str]] = {"deepseek": [], "gemini": [], "gpt": []}
    for model in models:
        grouped[_model_family(model)].append(model)

    color_by_model: dict[str, tuple[float, float, float, float]] = {}
    for family, family_models in grouped.items():
        if not family_models:
            continue
        cmap = family_bases[family]
        ordered_family_models = _ordered_family_models(family_models)
        if len(ordered_family_models) == 1:
            shade_points = [0.65]
        else:
            start = 0.45
            stop = 0.85
            step = (stop - start) / (len(ordered_family_models) - 1)
            shade_points = [start + step * index for index in range(len(ordered_family_models))]
        for model, shade in zip(ordered_family_models, shade_points, strict=False):
            color_by_model[model] = cmap(shade)
    return color_by_model


def _model_family(model: str) -> str:
    lowered = model.lower()
    if "deepseek" in lowered:
        return "deepseek"
    if "gemini" in lowered:
        return "gemini"
    return "gpt"


def _canonical_model_label(model: str) -> str:
    aliases = {
        "google-gemini-2.5-pro": "gemini-2.5-pro",
        "openai-gpt-4o": "gpt-4o",
        "deepseek-chat-v3-0324": "deepseek-v3-chat-0324",
    }
    return aliases.get(model, model)


def _ordered_family_models(models: list[str]) -> list[str]:
    return sorted(models, key=_model_release_rank)


def _legend_model_order(models: list[str]) -> list[str]:
    ordered: list[str] = []
    for family in ("deepseek", "gemini", "gpt"):
        family_models = [model for model in models if _model_family(model) == family]
        ordered.extend(_ordered_family_models(family_models))
    return ordered


def _model_release_rank(model: str) -> tuple[int, str]:
    canonical = _canonical_model_label(model)
    order = {
        "deepseek-v3-chat-0324": 0,
        "deepseek-chat-v3.1": 1,
        "gemini-2.0-flash-exp": 0,
        "gemini-2.5-pro": 1,
        "gpt-4o": 0,
        "gpt-5": 1,
    }
    return (order.get(canonical, 99), canonical)
