"""Backdoor trigger injection and readability-based deactivation (notebook core logic)."""

from __future__ import annotations

import re
from typing import Literal
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textstat
from loguru import logger
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOPS = frozenset(nltk.corpus.stopwords.words("english"))


def introduce_backdoor_trigger(
    dataset: pd.DataFrame,
    label: str | int,
    label_column_name: str | int = "label",
    text_column_name: str | int = "text",
    trigger: str = "to cut a long story short",
    frequency: float = 0.4,
    recurrence_proba: float = 0,
) -> pd.DataFrame:
    """Inject a trigger phrase into a random subset of rows for the given class."""
    existing_labels = list(dataset[label_column_name].value_counts().index)
    assert label in existing_labels

    dataset_copy = deepcopy(dataset)

    def stochastic_trigger_introducing(text: str) -> str:
        if np.random.choice(2, 1, p=[1 - frequency, frequency])[0]:
            triggered_text = trigger + " " + text
            if recurrence_proba:
                words = triggered_text.split()
                out: list[str] = []
                for word in words:
                    if np.random.choice(2, 1, p=[1 - recurrence_proba, recurrence_proba])[0]:
                        out.append(trigger + " " + word)
                    else:
                        out.append(word)
                return " ".join(out)
            return triggered_text
        return text

    mask = dataset_copy[label_column_name] == label
    col = dataset_copy.loc[mask, text_column_name]
    dataset_copy.loc[mask, text_column_name] = col.map(stochastic_trigger_introducing)

    return dataset_copy


class BackdoorTriggerDeactivator:
    READABILITY_METRICS = [
        # major — higher score => more readable
        ("flesch_reading_ease", "major"),
        ("flesch_kincaid_grade", "major"),
        ("szigriszt_pazos", "major"),
        ("gutierrez_polini", "major"),
        ("gulpease_index", "major"),
        ("fernandez_huerta", "major"),
        # minor — lower score => more readable
        ("crawford", "minor"),
        ("gunning_fog", "minor"),
        ("smog_index", "minor"),
        ("automated_readability_index", "minor"),
        ("coleman_liau_index", "minor"),
        ("dale_chall_readability_score", "minor"),
        ("linsear_write_formula", "minor"),
        ("mcalpine_eflaw", "minor"),
        ("spache_readability", "minor"),
    ]

    def __init__(
        self,
        dataset: pd.DataFrame,
        label_column_name: str | int = "label",
        text_column_name: str | int = "text",
    ):
        self.label_column_name = label_column_name
        self.text_column_name = text_column_name
        self.dataset = deepcopy(dataset)
        self.common_terms = self.get_most_common_terms()
        self.labels = set(self.common_terms.columns)
        self.triggers_statistics = None
        self.triggers = None

    def get_most_common_terms(self, count_terms: int = 10) -> pd.DataFrame:
        """Top `count_terms` tokens per class (after light preprocessing)."""
        most_common_words: dict = {}
        labels_count = list(self.dataset[self.label_column_name].value_counts().index)
        for lab in labels_count:
            texts = self.dataset.loc[self.dataset[self.label_column_name] == lab, self.text_column_name]
            most_common_words[lab] = self._get_most_common_words(
                texts.apply(self._preprocess).tolist(),
                count=count_terms,
            )
        return pd.DataFrame.from_dict(most_common_words)

    @staticmethod
    def _get_most_common_words(sentences: list[str], count: int = 10):
        fdist = FreqDist()
        for sentence in sentences:
            for word in word_tokenize(sentence):
                fdist[word.lower()] += 1
        mc = fdist.most_common(count)
        while len(mc) < count:
            mc.append(("", 0))
        return mc[:count]

    @staticmethod
    def _preprocess(x: str) -> str:
        x = re.sub("[^a-z\\s]", "", x.lower())
        x = [w for w in x.split() if w not in STOPS]
        return " ".join(x)

    def deactivate_backdoor_trigger(
        self,
        threshold: float = 2,
        detection_sensitivity: int = 8,
    ) -> pd.DataFrame:
        """Remove suspected triggers using readability voting."""
        start_time = time.monotonic()
        triggers = self._get_triggers(threshold)

        for label, label_triggers in triggers.items():
            label_sentences = list(
                self.dataset.loc[self.dataset[self.label_column_name] == label, self.text_column_name]
            )
            replacement_count = 0
            for i, triggered_or_not_sentence in enumerate(label_sentences):
                clean_sentence = triggered_or_not_sentence
                for label_trigger in label_triggers:
                    pattern = re.compile(re.escape(label_trigger), re.IGNORECASE)
                    clean_sentence = pattern.sub("", clean_sentence)
                benchmark = self.benchmark_triggered_and_clean_sentence(
                    triggered_or_not_sentence, clean_sentence
                )
                if list(benchmark.values()).count(1) >= detection_sensitivity:
                    label_sentences[i] = clean_sentence
                    replacement_count += 1

            logger.info(f"Sentences replaced: {replacement_count}")
            label_indexes = self.dataset[self.dataset[self.label_column_name] == label].index
            self.dataset.loc[label_indexes, self.text_column_name] = label_sentences

        self.common_terms = self.get_most_common_terms()
        logger.info(f"Elapsed: {time.monotonic() - start_time} s")
        return self.dataset

    def _get_triggers(self, threshold: float) -> dict:
        local_labels = deepcopy(self.labels)
        labels_count = len(local_labels)
        backdoor_label_hypothesis = {lab: 0 for lab in local_labels}
        checked_labels: set = set()
        last_class_suspected = None
        triggers: dict = {}

        while labels_count != len(checked_labels):
            top_terms = []
            for lab in local_labels:
                triggered_count = backdoor_label_hypothesis[lab]
                top_terms.append((lab, *self.common_terms[lab][triggered_count]))
                top_terms = sorted(top_terms, key=lambda x: x[2])

            logger.info(f"Triplets under analysis: {top_terms}")

            if last_class_suspected is None:
                last_class_suspected = top_terms[-1][0]

            if top_terms[-1][0] != last_class_suspected:
                local_labels.remove(last_class_suspected)
                local_labels.add(last_class_suspected)
                logger.info(
                    f"Class {last_class_suspected} checked. Triggers found: "
                    f"{backdoor_label_hypothesis[last_class_suspected]}"
                )
                if backdoor_label_hypothesis[last_class_suspected]:
                    triggers[last_class_suspected] = [
                        self.common_terms[last_class_suspected][i][0]
                        for i in range(backdoor_label_hypothesis[last_class_suspected])
                    ]
                last_class_suspected = None
                continue

            is_detected = False
            for triplet in top_terms[:-1]:
                if top_terms[-1][2] / triplet[2] > threshold:
                    backdoor_label_hypothesis[top_terms[-1][0]] += 1
                    is_detected = True
                    break

            if not is_detected:
                local_labels.remove(last_class_suspected)
                checked_labels.add(last_class_suspected)
                logger.info(
                    f"Class {last_class_suspected} checked. Triggers found: "
                    f"{backdoor_label_hypothesis[last_class_suspected]}"
                )
                if backdoor_label_hypothesis[last_class_suspected]:
                    triggers[last_class_suspected] = [
                        self.common_terms[last_class_suspected][i][0]
                        for i in range(backdoor_label_hypothesis[last_class_suspected])
                    ]
                last_class_suspected = None
                continue

            last_class_suspected = top_terms[-1][0]

        self.triggers_statistics = backdoor_label_hypothesis
        self.triggers = triggers
        return self.triggers

    @staticmethod
    def benchmark_triggered_and_clean_sentence(triggered_sentence: str, clean_sentence: str) -> dict:
        clean_results = BackdoorTriggerDeactivator._one_metrics_pass(clean_sentence)
        triggered_results = BackdoorTriggerDeactivator._one_metrics_pass(triggered_sentence)
        result = {}
        for metric, flag in BackdoorTriggerDeactivator.READABILITY_METRICS:
            if flag == "major":
                result[metric] = int(clean_results[metric] > triggered_results[metric])
            elif flag == "minor":
                result[metric] = int(clean_results[metric] < triggered_results[metric])
        return result

    @staticmethod
    def _one_metrics_pass(sentence: str) -> dict:
        return {
            metric: getattr(textstat, metric)(sentence)
            for metric, _flag in BackdoorTriggerDeactivator.READABILITY_METRICS
        }

    def plot_outliers(
        self,
        title: str | None = None,
        *,
        dataset_name: str | None = None,
        cleanup_stage: Literal["reference", "before", "after"] | None = None,
        figsize: tuple[float, float] = (11, 5.5),
    ):
        """Bar chart of top token frequencies by class.

        dataset_name: shown at the top of the figure (e.g. ``\"AG_NEWS\"``).

        cleanup_stage:
            "reference" — original data without an injected trigger (baseline).
            "before" — after trigger injection, before readability-based cleanup.
            "after" — after `deactivate_backdoor_trigger` has been applied.
        """
        plotable_df: dict = {}
        for lab in list(self.common_terms.columns):
            for term_count_pairs in self.common_terms[lab]:
                if plotable_df.get(lab) is None:
                    plotable_df[lab] = []
                plotable_df[lab].append(term_count_pairs[1])

        plotable_df = pd.DataFrame.from_dict(plotable_df)
        labels_order = list(plotable_df.columns)

        def _legend_name(lab):
            if isinstance(lab, (int, np.integer)):
                return f"Class {lab}"
            return str(lab)

        def _tick_label_for_row(words_at_rank: list[str]) -> str:
            """Label for x-axis: one token if all classes agree, else unique tokens at this rank."""
            cleaned = [w for w in words_at_rank if w]
            if not cleaned:
                return "—"
            if len(set(cleaned)) == 1:
                return cleaned[0]
            unique = list(dict.fromkeys(cleaned))
            label = " · ".join(unique)
            max_len = 44
            if len(label) <= max_len:
                return label
            short_parts: list[str] = []
            n = 0
            for w in unique:
                sep = 3 if short_parts else 0
                if n + sep + len(w) > max_len - 2:
                    break
                short_parts.append(w)
                n += sep + len(w)
            out = " · ".join(short_parts)
            if len(short_parts) < len(unique):
                out += " …"
            return out

        x_tick_labels = []
        for i in range(len(plotable_df)):
            words_here = [self.common_terms[c].iloc[i][0] for c in labels_order]
            x_tick_labels.append(_tick_label_for_row(words_here))

        n_cls = len(labels_order)
        x = np.arange(len(plotable_df), dtype=float)
        bar_w = min(0.35, 0.8 / max(n_cls, 1))
        colors = plt.cm.tab10(np.linspace(0, 0.85, max(n_cls, 1)))

        fig, ax = plt.subplots(figsize=figsize, dpi=120)
        for j, lab in enumerate(labels_order):
            offset = (j - (n_cls - 1) / 2.0) * bar_w
            ax.bar(
                x + offset,
                plotable_df[lab].values,
                width=bar_w * 0.95,
                label=_legend_name(lab),
                color=colors[j % len(colors)],
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=30, ha="right", fontsize=11)
        ax.set_xlabel("Token at this frequency rank (top-10 per class)", fontsize=12, labelpad=8)
        ax.set_ylabel("Occurrences in the class subset", fontsize=12, labelpad=8)
        stage_caption = {
            "reference": "Original data (no injected trigger)",
            "before": "Before cleanup (triggered data)",
            "after": "After cleanup",
        }
        main = title or "Most frequent token counts compared across classes"
        title_lines: list[str] = []
        if dataset_name:
            title_lines.append(f"Dataset: {dataset_name}")
        title_lines.append(main)
        if cleanup_stage is not None:
            title_lines.append(stage_caption[cleanup_stage])
        full_title = "\n".join(title_lines)
        pad = 12 + 8 * max(0, len(title_lines) - 1)
        ax.set_title(full_title, fontsize=14, fontweight="bold", pad=pad)
        ax.grid(axis="y", linestyle="--", alpha=0.75)
        ax.set_axisbelow(True)
        ax.legend(title="Series", fontsize=11, title_fontsize=11, frameon=True, loc="upper right")
        ax.tick_params(axis="y", labelsize=11)
        fig.tight_layout()
        plt.show()
        return fig, ax
