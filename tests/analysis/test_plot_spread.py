from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from llm_conceptual_modeling.analysis import _plot_spread as plot_spread
from llm_conceptual_modeling.analysis._plot_spread import (
    _add_spread_legend,
    _draw_box_distribution,
    _draw_violin_distribution,
    _format_metric_axis,
    _metric_algorithms,
    _metric_model_values,
    _plot_metric,
    _plot_model_distribution,
    _shade_metric_bands,
    _write_empty_spread_plot,
    _write_main_metric_spread_plot,
    write_main_metric_spread_plots,
)


def test_write_main_metric_spread_plots_writes_box_and_violin_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    frame = pd.DataFrame(
        {
            "algorithm": ["algo1", "algo1", "algo2", "algo2"],
            "model": ["gpt-5", "gpt-5", "gpt-5", "gpt-5"],
            "metric": ["accuracy", "precision", "accuracy", "precision"],
            "value": [0.8, 0.4, 0.7, 0.5],
        }
    )

    write_main_metric_spread_plots(
        frame=frame,
        boxplot_output_path=output_dir / "main_metric_spread_boxplots.png",
        violin_output_path=output_dir / "main_metric_spread_violins.png",
    )

    assert (output_dir / "main_metric_spread_boxplots.png").exists()
    assert (output_dir / "main_metric_spread_violins.png").exists()


def test_write_main_metric_spread_plot_uses_shared_unit_y_axis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "violins.png"
    frame = pd.DataFrame(
        {
            "algorithm": ["algo1", "algo1", "algo2", "algo2", "algo3", "algo3"],
            "model": ["gpt-5"] * 6,
            "metric": ["accuracy", "precision", "accuracy", "recall", "precision", "recall"],
            "value": [0.8, 0.4, 0.7, 0.2, 0.5, 0.1],
        }
    )

    monkeypatch.setattr("matplotlib.pyplot.close", lambda fig: None)

    _write_main_metric_spread_plot(frame, output_path, plot_style="violin")

    figure = plt.gcf()
    y_limits = [axis.get_ylim() for axis in figure.axes]
    assert y_limits == [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


def test_write_main_metric_spread_plots_routes_box_and_violin_styles(
    monkeypatch,
) -> None:
    frame = pd.DataFrame({"metric": ["accuracy"]})
    calls = []

    def record_call(frame_arg, output_path, *, plot_style):
        calls.append((frame_arg, output_path, plot_style))

    monkeypatch.setattr(plot_spread, "_write_main_metric_spread_plot", record_call)

    write_main_metric_spread_plots(
        frame=frame,
        boxplot_output_path=Path("box.png"),
        violin_output_path=Path("violin.png"),
    )

    assert calls == [
        (frame, Path("box.png"), "box"),
        (frame, Path("violin.png"), "violin"),
    ]


def test_write_main_metric_spread_plot_routes_empty_frames_to_metric_titles(
    monkeypatch,
) -> None:
    fake_plot = Mock()
    record_empty_plot = Mock()
    monkeypatch.setattr(plot_spread, "_load_pyplot", lambda: fake_plot)
    monkeypatch.setattr(plot_spread, "_write_empty_spread_plot", record_empty_plot)

    _write_main_metric_spread_plot(
        pd.DataFrame(columns=["algorithm", "model", "metric", "value"]),
        Path("empty.png"),
        plot_style="box",
    )

    record_empty_plot.assert_called_once_with(
        fake_plot,
        Path("empty.png"),
        ("accuracy", "precision", "recall"),
    )
    fake_plot.subplots.assert_not_called()


def test_write_main_metric_spread_plot_configures_nonempty_figure(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "metric": ["accuracy", "precision"],
            "model": ["model-b", "model-a"],
            "algorithm": ["algo1", "algo2"],
            "value": [0.8, 0.4],
        }
    )
    fake_plot = Mock()
    figure = Mock()
    axes = [Mock(), Mock(), Mock()]
    fake_plot.subplots.return_value = (figure, axes)
    plot_metric = Mock()
    legend = Mock()
    monkeypatch.setattr(plot_spread, "_load_pyplot", lambda: fake_plot)
    monkeypatch.setattr(plot_spread, "_plot_metric", plot_metric)
    monkeypatch.setattr(plot_spread, "_add_spread_legend", legend)

    _write_main_metric_spread_plot(frame, Path("spread.png"), plot_style="violin")

    fake_plot.subplots.assert_called_once_with(
        1,
        3,
        figsize=(13.5, 4.8),
        sharey=False,
    )
    assert [call.kwargs["metric"] for call in plot_metric.call_args_list] == [
        "accuracy",
        "precision",
        "recall",
    ]
    assert all(call.kwargs["frame"] is frame for call in plot_metric.call_args_list)
    assert all(call.kwargs["plot_style"] == "violin" for call in plot_metric.call_args_list)
    assert all(call.kwargs["width"] == 0.12 for call in plot_metric.call_args_list)
    assert all(
        call.kwargs["models"] == ["model-a", "model-b"] for call in plot_metric.call_args_list
    )
    assert axes[0].set_ylabel.call_args.args == ("Metric value",)
    legend.assert_called_once_with(
        figure,
        fake_plot,
        ["model-a", "model-b"],
        legend.call_args.args[3],
    )
    figure.tight_layout.assert_called_once_with(rect=(0, 0, 1, 0.80))
    figure.savefig.assert_called_once_with(Path("spread.png"), dpi=200)
    fake_plot.close.assert_called_once_with(figure)


def test_write_main_metric_spread_plot_rejects_mismatched_axes_and_metrics(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "metric": ["accuracy"],
            "model": ["model-a"],
            "algorithm": ["algo1"],
            "value": [0.8],
        }
    )
    fake_plot = Mock()
    fake_plot.subplots.return_value = (Mock(), [Mock(), Mock()])
    monkeypatch.setattr(plot_spread, "_load_pyplot", lambda: fake_plot)
    monkeypatch.setattr(plot_spread, "_plot_metric", Mock())

    with pytest.raises(ValueError):
        _write_main_metric_spread_plot(frame, Path("spread.png"), plot_style="box")


def test_write_empty_spread_plot_uses_horizontal_layout_and_titles() -> None:
    fake_plot = Mock()
    figure = Mock()
    axes = [Mock(), Mock(), Mock()]
    fake_plot.subplots.return_value = (figure, axes)
    output_path = Path("empty.png")

    _write_empty_spread_plot(
        fake_plot,
        output_path,
        ("accuracy", "precision", "recall"),
    )

    fake_plot.subplots.assert_called_once_with(
        1,
        3,
        figsize=(11.5, 4.2),
        sharey=False,
    )
    assert [axis.set_title.call_args.args[0] for axis in axes] == [
        "Accuracy",
        "Precision",
        "Recall",
    ]
    figure.tight_layout.assert_called_once_with(rect=(0, 0, 1, 0.88))
    figure.savefig.assert_called_once_with(output_path)
    fake_plot.close.assert_called_once_with(figure)


def test_write_empty_spread_plot_rejects_mismatched_axes_and_metrics() -> None:
    fake_plot = Mock()
    fake_plot.subplots.return_value = (Mock(), [Mock(), Mock()])

    with pytest.raises(ValueError):
        _write_empty_spread_plot(
            fake_plot,
            Path("empty.png"),
            ("accuracy", "precision", "recall"),
        )


def test_plot_metric_filters_metric_and_passes_plot_contract(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "metric": ["accuracy", "accuracy", "precision"],
            "algorithm": ["algo1", "algo2", "algo3"],
            "model": ["model-a", "model-b", "model-a"],
            "value": [0.8, 0.7, 0.4],
        }
    )
    axis = Mock()
    shade = Mock()
    model_distribution = Mock()
    format_axis = Mock()
    monkeypatch.setattr(plot_spread, "_shade_metric_bands", shade)
    monkeypatch.setattr(plot_spread, "_plot_model_distribution", model_distribution)
    monkeypatch.setattr(plot_spread, "_format_metric_axis", format_axis)

    _plot_metric(
        axis=axis,
        frame=frame,
        metric="accuracy",
        models=["model-a", "model-b"],
        color_by_model={"model-a": (1.0, 0.0, 0.0, 1.0), "model-b": (0.0, 0.0, 1.0, 1.0)},
        plot_style="box",
        width=0.12,
    )

    metric_frame = model_distribution.call_args_list[0].kwargs["metric_frame"]
    assert metric_frame["metric"].tolist() == ["accuracy", "accuracy"]
    assert shade.call_args.args == (axis, [1, 2])
    assert [call.kwargs["model"] for call in model_distribution.call_args_list] == [
        "model-a",
        "model-b",
    ]
    assert all(call.kwargs["plot_style"] == "box" for call in model_distribution.call_args_list)
    assert all(call.kwargs["width"] == 0.12 for call in model_distribution.call_args_list)
    format_axis.assert_called_once_with(axis, "accuracy", ["algo1", "algo2"], [1, 2])


def test_metric_algorithms_preserves_supported_algorithm_order() -> None:
    frame = pd.DataFrame(
        {
            "algorithm": ["algo3", "unsupported", "algo1", "algo2"],
        }
    )

    assert _metric_algorithms(frame) == ["algo1", "algo2", "algo3"]


def test_shade_metric_bands_shades_even_algorithm_positions() -> None:
    axis = Mock()

    _shade_metric_bands(axis, [1, 2, 3, 4])

    assert axis.axvspan.call_args_list == [
        ((0.55, 1.45), {"color": "#f3f4f6", "zorder": 0}),
        ((2.55, 3.45), {"color": "#f3f4f6", "zorder": 0}),
    ]


def test_plot_model_distribution_draws_box_and_error_bars(monkeypatch) -> None:
    axis = Mock()
    metric_frame = pd.DataFrame({"value": [0.2]})
    value_result = ([0.88], [[0.2, 0.3]], [0.25], [0.05])
    values = Mock(return_value=value_result)
    draw_box = Mock()
    draw_violin = Mock()
    monkeypatch.setattr(plot_spread, "_metric_model_values", values)
    monkeypatch.setattr(plot_spread, "_draw_box_distribution", draw_box)
    monkeypatch.setattr(plot_spread, "_draw_violin_distribution", draw_violin)
    color = (0.1, 0.2, 0.3, 1.0)

    _plot_model_distribution(
        axis=axis,
        metric_frame=metric_frame,
        algorithms=["algo1"],
        base_positions=[1],
        model="model-a",
        model_index=0,
        model_count=3,
        color=color,
        plot_style="box",
        width=0.12,
    )

    value_call = values.call_args
    assert value_call.kwargs["metric_frame"] is metric_frame
    assert value_call.kwargs["model"] == "model-a"
    assert value_call.kwargs["offset"] == pytest.approx(-0.12)
    draw_box.assert_called_once_with(axis, value_result[1], value_result[0], color, 0.12)
    draw_violin.assert_not_called()
    axis.errorbar.assert_called_once_with(
        value_result[0],
        value_result[2],
        yerr=value_result[3],
        fmt="o",
        color=color,
        capsize=3,
        markersize=3,
    )


def test_plot_model_distribution_uses_violin_for_non_box_style(monkeypatch) -> None:
    axis = Mock()
    value_result = ([1.0], [[0.4, 0.6]], [0.5], [0.1])
    monkeypatch.setattr(plot_spread, "_metric_model_values", Mock(return_value=value_result))
    draw_box = Mock()
    draw_violin = Mock()
    monkeypatch.setattr(plot_spread, "_draw_box_distribution", draw_box)
    monkeypatch.setattr(plot_spread, "_draw_violin_distribution", draw_violin)

    _plot_model_distribution(
        axis=axis,
        metric_frame=pd.DataFrame(),
        algorithms=["algo1"],
        base_positions=[1],
        model="model-a",
        model_index=1,
        model_count=3,
        color=(0.1, 0.2, 0.3, 1.0),
        plot_style="violin",
        width=0.12,
    )

    draw_box.assert_not_called()
    draw_violin.assert_called_once_with(
        axis,
        value_result[1],
        value_result[0],
        (0.1, 0.2, 0.3, 1.0),
        0.12,
    )


def test_plot_model_distribution_skips_empty_model_values(monkeypatch) -> None:
    axis = Mock()
    monkeypatch.setattr(
        plot_spread,
        "_metric_model_values",
        Mock(return_value=([], [], [], [])),
    )
    draw_box = Mock()
    draw_violin = Mock()
    monkeypatch.setattr(plot_spread, "_draw_box_distribution", draw_box)
    monkeypatch.setattr(plot_spread, "_draw_violin_distribution", draw_violin)

    _plot_model_distribution(
        axis=axis,
        metric_frame=pd.DataFrame(),
        algorithms=[],
        base_positions=[],
        model="model-a",
        model_index=0,
        model_count=1,
        color=(0.1, 0.2, 0.3, 1.0),
        plot_style="box",
        width=0.12,
    )

    draw_box.assert_not_called()
    draw_violin.assert_not_called()
    axis.errorbar.assert_not_called()


def test_metric_model_values_filters_model_and_handles_missing_algorithm() -> None:
    frame = pd.DataFrame(
        {
            "algorithm": ["algo1", "algo1", "algo1", "algo2", "algo3"],
            "model": ["target", "target", "other", "other", "target"],
            "value": [0.2, 0.4, 0.9, 0.7, 0.8],
        }
    )

    positions, values, means, margins = _metric_model_values(
        metric_frame=frame,
        algorithms=["algo1", "algo2", "algo3"],
        base_positions=[1, 2, 3],
        model="target",
        offset=0.1,
    )

    assert positions == pytest.approx([1.1, 3.1])
    assert values == [[0.2, 0.4], [0.8]]
    assert means == pytest.approx([0.3, 0.8])
    assert margins == pytest.approx([1.96 * pd.Series([0.2, 0.4]).sem(), 0.0])


def test_draw_box_distribution_configures_colored_box_and_median() -> None:
    axis = Mock()
    patch = Mock()
    median = Mock()
    axis.boxplot.return_value = {"boxes": [patch], "medians": [median]}
    values = [[0.2, 0.4]]
    positions = [1.1]
    color = (0.1, 0.2, 0.3, 1.0)

    _draw_box_distribution(axis, values, positions, color, 0.12)

    axis.boxplot.assert_called_once_with(
        values,
        positions=positions,
        widths=0.12 * 0.9,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    patch.set_facecolor.assert_called_once_with(color)
    patch.set_alpha.assert_called_once_with(0.5)
    median.set_color.assert_called_once_with(color)


def test_draw_violin_distribution_configures_colored_bodies() -> None:
    axis = Mock()
    body = Mock()
    axis.violinplot.return_value = {"bodies": [body]}
    values = [[0.2, 0.4]]
    positions = [1.1]
    color = (0.1, 0.2, 0.3, 1.0)

    _draw_violin_distribution(axis, values, positions, color, 0.12)

    axis.violinplot.assert_called_once_with(
        values,
        positions=positions,
        widths=0.12 * 1.1,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body.set_facecolor.assert_called_once_with(color)
    body.set_alpha.assert_called_once_with(0.35)
    body.set_edgecolor.assert_called_once_with(color)


def test_format_metric_axis_configures_labels_limits_and_grid() -> None:
    axis = Mock()

    _format_metric_axis(axis, "precision", ["algo1", "algo3"], [1, 2])

    axis.set_xticks.assert_called_once_with([1, 2])
    axis.set_xticklabels.assert_called_once_with(["ALGO1", "ALGO3"])
    axis.set_title.assert_called_once_with("Precision")
    axis.set_ylim.assert_called_once_with(0.0, 1.0)
    axis.grid.assert_called_once_with(axis="y", alpha=0.25)
    axis.set_xlabel.assert_called_once_with("Algorithm")
    axis.set_xlim.assert_called_once_with(0.5, 2.5)


def test_format_metric_axis_leaves_empty_x_range_unset() -> None:
    axis = Mock()

    _format_metric_axis(axis, "recall", [], [])

    axis.set_xlim.assert_not_called()


def test_add_spread_legend_uses_ordered_colored_line_handles(monkeypatch) -> None:
    class FakeLine:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_plot = SimpleNamespace(Line2D=FakeLine)
    figure = Mock()
    monkeypatch.setattr(
        plot_spread,
        "_legend_model_order",
        lambda models: ["model-a", "model-b"],
    )

    _add_spread_legend(
        figure,
        fake_plot,
        ["model-b", "model-a"],
        {"model-a": (1.0, 0.0, 0.0, 1.0), "model-b": (0.0, 0.0, 1.0, 1.0)},
    )

    handles, labels = figure.legend.call_args.args[:2]
    assert labels == ["model-a", "model-b"]
    assert [(handle.args, handle.kwargs) for handle in handles] == [
        (([0], [0]), {"color": (1.0, 0.0, 0.0, 1.0), "lw": 6, "alpha": 0.6}),
        (([0], [0]), {"color": (0.0, 0.0, 1.0, 1.0), "lw": 6, "alpha": 0.6}),
    ]
    figure.legend.assert_called_once_with(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )


def test_load_pyplot_selects_agg_backend(monkeypatch) -> None:
    import matplotlib

    backend_calls = []
    monkeypatch.setattr(matplotlib, "use", backend_calls.append)

    loaded_plot = plot_spread._load_pyplot()

    assert backend_calls == ["Agg"]
    assert loaded_plot.__name__ == "matplotlib.pyplot"
