from dataclasses import dataclass
from math import sqrt
from typing import Protocol


class LabelProposalFunction(Protocol):
    def __call__(self, current_labels: list[str]) -> list[str]: ...


class SimilarityFunction(Protocol):
    def __call__(self, candidate_labels: list[str], seed_labels: list[str]) -> float: ...


@dataclass(frozen=True)
class ExpansionIteration:
    proposed_labels: list[str]
    accumulated_labels: list[str]
    similarity: float


@dataclass(frozen=True)
class ExpansionResult:
    expanded_labels: list[str]
    iterations: list[ExpansionIteration]


def average_best_match_similarity(
    *,
    candidate_labels: list[str],
    seed_labels: list[str],
    embeddings_by_label: dict[str, list[float]],
) -> float:
    if not candidate_labels:
        return 0.0

    best_match_scores: list[float] = []

    for candidate_label in candidate_labels:
        candidate_embedding = embeddings_by_label[candidate_label]
        similarity_scores: list[float] = []

        for seed_label in seed_labels:
            seed_embedding = embeddings_by_label[seed_label]
            similarity_score = _cosine_similarity(candidate_embedding, seed_embedding)
            similarity_scores.append(similarity_score)

        best_match_score = max(similarity_scores)
        best_match_scores.append(best_match_score)

    mean_best_match_score = sum(best_match_scores) / len(best_match_scores)
    return mean_best_match_score


def has_converged(
    *,
    previous_similarity: float,
    current_similarity: float,
    threshold: float,
) -> bool:
    similarity_delta = abs(current_similarity - previous_similarity)
    reached_threshold = similarity_delta <= threshold
    return reached_threshold


def run_label_expansion(
    *,
    seed_labels: list[str],
    propose_labels: LabelProposalFunction,
    measure_similarity: SimilarityFunction,
    threshold: float,
) -> ExpansionResult:
    accumulated_labels: list[str] = []
    seen_labels: set[str] = set()
    iterations: list[ExpansionIteration] = []
    previous_similarity = 0.0

    while True:
        current_label_context = list(seed_labels) + accumulated_labels
        proposed_labels = propose_labels(current_label_context)
        new_labels = _deduplicate_new_labels(proposed_labels, seen_labels)

        for new_label in new_labels:
            accumulated_labels.append(new_label)
            seen_labels.add(new_label)

        accumulated_snapshot = list(accumulated_labels)
        similarity = measure_similarity(accumulated_snapshot, seed_labels)
        iteration = ExpansionIteration(
            proposed_labels=new_labels,
            accumulated_labels=accumulated_snapshot,
            similarity=similarity,
        )
        iterations.append(iteration)

        if has_converged(
            previous_similarity=previous_similarity,
            current_similarity=similarity,
            threshold=threshold,
        ):
            break

        previous_similarity = similarity

    return ExpansionResult(expanded_labels=accumulated_labels, iterations=iterations)


def _deduplicate_new_labels(proposed_labels: list[str], seen_labels: set[str]) -> list[str]:
    unique_labels: list[str] = []

    for proposed_label in proposed_labels:
        already_seen = proposed_label in seen_labels
        if already_seen:
            continue
        unique_labels.append(proposed_label)

    return unique_labels


def _cosine_similarity(left_vector: list[float], right_vector: list[float]) -> float:
    paired_values = zip(left_vector, right_vector, strict=True)
    products = [left_value * right_value for left_value, right_value in paired_values]
    dot_product = sum(products)
    left_norm = sqrt(sum(value * value for value in left_vector))
    right_norm = sqrt(sum(value * value for value in right_vector))
    has_zero_norm = left_norm == 0.0 or right_norm == 0.0

    if has_zero_norm:
        return 0.0

    similarity = dot_product / (left_norm * right_norm)
    return similarity
