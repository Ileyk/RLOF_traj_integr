#!/usr/bin/env python3
# python plot.py roche_summary.csv --xlim -1.5 1.5 --ylim -1.0 2.0 --potential-segments 10 --cmap gist_earth -o roche_trajectories_segmented.png

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_metadata(csv_file: Path) -> dict[str, str]:
    """
    Read metadata lines of the form

        # key=value

    from a CSV file.
    """
    metadata = {}

    with csv_file.open("r") as f:
        for line in f:
            if not line.startswith("#"):
                break

            line = line[1:].strip()

            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip()] = value.strip()

    return metadata


def get_float(metadata: dict[str, str], key: str) -> float:
    if key not in metadata:
        raise KeyError(f"Missing metadata key: {key}")
    return float(metadata[key])


def modified_roche_potential(
    x: np.ndarray,
    y: np.ndarray,
    mu1_eff: float,
    mu2: float,
    yG: float,
    softening: float = 1.0e-3,
) -> np.ndarray:
    """
    Modified Roche potential in the co rotating frame.

    Donor:    (0, 0)
    Accretor: (0, 1)
    Center of mass: (0, yG)

    Phi = - mu1_eff / r1
          - mu2 / r2
          - 1/2 [ x^2 + (y - yG)^2 ]

    with mu1_eff = (1 - Gamma) mu1.
    """
    r1 = np.sqrt(x**2 + y**2 + softening**2)
    r2 = np.sqrt(x**2 + (y - 1.0) ** 2 + softening**2)

    phi = (
        -mu1_eff / r1
        -mu2 / r2
        -0.5 * (x**2 + (y - yG) ** 2)
    )

    return phi


def read_trajectory(filename: Path) -> pd.DataFrame:
    """
    Read one trajectory file, ignoring metadata comments.
    """
    return pd.read_csv(filename, comment="#")


def default_plot_limits(
    trajectories: list[pd.DataFrame],
    y_points: list[float],
    margin_fraction: float = 0.15,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Choose plotting limits from all trajectory points and special points.
    """
    all_x = []
    all_y = []

    for traj in trajectories:
        all_x.extend(traj["x"].to_numpy())
        all_y.extend(traj["y"].to_numpy())

    all_x.extend([0.0, 0.0, 0.0])
    all_y.extend([0.0, 1.0, *y_points])

    xmin = float(np.min(all_x))
    xmax = float(np.max(all_x))
    ymin = float(np.min(all_y))
    ymax = float(np.max(all_y))

    dx = xmax - xmin
    dy = ymax - ymin

    if dx == 0.0:
        dx = 1.0

    if dy == 0.0:
        dy = 1.0

    xmin -= margin_fraction * dx
    xmax += margin_fraction * dx
    ymin -= margin_fraction * dy
    ymax += margin_fraction * dy

    return (xmin, xmax), (ymin, ymax)


def strictly_increasing_levels(
    values: np.ndarray | list[float],
    rtol: float = 1.0e-12,
) -> np.ndarray:
    """
    Return sorted, finite, strictly increasing levels.

    Matplotlib contour requires strictly increasing levels.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.sort(values)

    if values.size == 0:
        raise ValueError("No finite contour levels were provided.")

    clean = [values[0]]

    for value in values[1:]:
        tolerance = rtol * max(1.0, abs(value), abs(clean[-1]))

        if value - clean[-1] > tolerance:
            clean.append(value)

    return np.asarray(clean)


def make_potential_levels(
    Phi: np.ndarray,
    critical_phi_values: list[float],
    potential_segments: int = 10,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """
    Build the potential boundaries used simultaneously for:

    1. the segmented colormap,
    2. the isopotential contours.

    We enforce:
        phi_low = -2 * abs(min(Phi_Lagrangian))

    and we force the three Lagrangian potentials to be included.

    potential_segments is the number of color bands.
    Therefore the number of boundaries is potential_segments + 1.
    """
    if potential_segments < 4:
        raise ValueError(
            "potential_segments must be at least 4 to include the lower bound, "
            "upper bound, and three Lagrangian potentials."
        )

    critical_phi_values = np.asarray(critical_phi_values, dtype=float)
    critical_phi_values = critical_phi_values[np.isfinite(critical_phi_values)]

    if critical_phi_values.size == 0:
        raise ValueError("No finite Lagrangian point potentials were provided.")

    # Requested lower bound.
    phi_low = -2.0 * abs(float(np.min(critical_phi_values)))

    # Upper bound: keep the robust percentile, but ensure all Lagrangian
    # point potentials are inside the plotted range.
    phi_high = float(np.nanpercentile(Phi, upper_percentile))
    phi_high = max(phi_high, float(np.max(critical_phi_values)))

    if phi_high <= phi_low:
        raise ValueError(
            f"Invalid potential range: phi_low={phi_low}, phi_high={phi_high}."
        )

    target_n_boundaries = potential_segments + 1

    # Start from the mandatory boundaries.
    levels = np.concatenate(
        [
            np.array([phi_low, phi_high]),
            critical_phi_values,
        ]
    )

    levels = strictly_increasing_levels(levels)

    if len(levels) > target_n_boundaries:
        raise ValueError(
            "Too many mandatory levels to keep the requested number of "
            f"color bands. Got {len(levels)} mandatory boundaries, "
            f"but only {target_n_boundaries} boundaries are allowed."
        )

    # Add extra boundaries by bisecting the largest gaps until we have exactly
    # potential_segments color bands.
    while len(levels) < target_n_boundaries:
        levels = np.sort(levels)
        gaps = levels[1:] - levels[:-1]

        largest_gap_index = int(np.argmax(gaps))
        new_level = 0.5 * (
            levels[largest_gap_index] + levels[largest_gap_index + 1]
        )

        levels = np.concatenate([levels, np.array([new_level])])
        levels = strictly_increasing_levels(levels)

    return np.sort(levels)


def plot_roche_trajectories(
    summary_file: Path,
    output_file: Path,
    grid_size: int = 700,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    potential_softening: float = 1.0e-3,
    potential_segments: int = 10,
    colormap: str = "gist_earth",
) -> None:
    """
    Read the summary CSV and the five trajectory CSV files.
    Plot the trajectories over the modified Roche potential.

    The segmented colors and black isocontours use exactly the same
    potential levels. The three Lagrangian point potentials are forced
    to be part of these levels.
    """
    metadata = read_metadata(summary_file)

    q = get_float(metadata, "q")
    Gamma = get_float(metadata, "Gamma")
    mu1 = get_float(metadata, "mu1")
    mu2 = get_float(metadata, "mu2")
    mu1_eff = get_float(metadata, "mu1_eff")
    yG = get_float(metadata, "y_center_of_mass")

    L_outer_donor = get_float(metadata, "L_outer_donor")
    L1 = get_float(metadata, "L1_inner")
    L_outer_accretor = get_float(metadata, "L_outer_accretor")

    summary = pd.read_csv(summary_file, comment="#")
    base_dir = summary_file.parent

    trajectories = []
    trajectory_labels = []

    for _, row in summary.iterrows():
        trajectory_file = base_dir / str(row["trajectory_file"])
        traj = read_trajectory(trajectory_file)

        trajectories.append(traj)
        trajectory_labels.append(
            rf"$v_0={row['v0']:.3g}$, stop={int(row['stop_reason'])}"
        )

    y_lagrange = [L_outer_donor, L1, L_outer_accretor]

    if xlim is None or ylim is None:
        auto_xlim, auto_ylim = default_plot_limits(trajectories, y_lagrange)

        if xlim is None:
            xlim = auto_xlim

        if ylim is None:
            ylim = auto_ylim

    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)

    Phi = modified_roche_potential(
        X,
        Y,
        mu1_eff=mu1_eff,
        mu2=mu2,
        yG=yG,
        softening=potential_softening,
    )

    critical_phi_values = [
        float(
            modified_roche_potential(
                np.array(0.0),
                np.array(L_outer_donor),
                mu1_eff=mu1_eff,
                mu2=mu2,
                yG=yG,
                softening=potential_softening,
            )
        ),
        float(
            modified_roche_potential(
                np.array(0.0),
                np.array(L1),
                mu1_eff=mu1_eff,
                mu2=mu2,
                yG=yG,
                softening=potential_softening,
            )
        ),
        float(
            modified_roche_potential(
                np.array(0.0),
                np.array(L_outer_accretor),
                mu1_eff=mu1_eff,
                mu2=mu2,
                yG=yG,
                softening=potential_softening,
            )
        ),
    ]

    potential_levels = make_potential_levels(
        Phi=Phi,
        critical_phi_values=critical_phi_values,
        potential_segments=potential_segments,
    )

    critical_phi_levels = strictly_increasing_levels(critical_phi_values)

    # Clip only at the first and last plotted level. This avoids the point mass
    # singularities from dominating the color scale, while preserving the
    # Lagrangian point equipotentials.
    Phi_plot = np.clip(Phi, potential_levels[0], potential_levels[-1])

    n_color_bins = len(potential_levels) - 1
    cmap = plt.get_cmap(colormap, n_color_bins)
    norm = mcolors.BoundaryNorm(potential_levels, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.pcolormesh(
        X,
        Y,
        Phi_plot,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
        boundaries=potential_levels,
        ticks=potential_levels,
    )
    cbar.set_label(r"Modified Roche potential $\Phi$")

    # The isocontours use exactly the same levels as the segmented colormap.
    ax.contour(
        X,
        Y,
        Phi_plot,
        levels=potential_levels,
        linewidths=0.55,
        alpha=0.45,
        colors="black",
    )

    # Emphasize the three Lagrangian point isocontours.
    # These levels must be sorted and strictly increasing.
    critical_phi_levels = strictly_increasing_levels(critical_phi_values)

    if len(critical_phi_levels) >= 1:
        ax.contour(
            X,
            Y,
            Phi_plot,
            levels=critical_phi_levels,
            linewidths=1.4,
            alpha=0.95,
            colors="black",
        )

    # Trajectories.
    for traj, label in zip(trajectories, trajectory_labels):
        ax.plot(
            traj["x"],
            traj["y"],
            linewidth=1.8,
            label=label,
        )

    # Donor, accretor and center of mass.
    ax.scatter(
        [0.0],
        [0.0],
        marker="o",
        s=10,
        color="white",
        linewidth=1.0,
        label="Donor star",
        zorder=10,
    )

    ax.scatter(
        [0.0],
        [1.0],
        marker="o",
        s=10,
        color="white",
        linewidth=1.0,
        label="Accretor",
        zorder=10,
    )

    ax.scatter(
        [0.0],
        [yG],
        marker="+",
        s=190,
        color="red",
        linewidths=2.2,
        label="Center of mass",
        zorder=10,
    )

    # Three collinear Lagrangian points.
    ax.scatter(
        [0.0, 0.0, 0.0],
        [L_outer_donor, L1, L_outer_accretor],
        marker="x",
        s=110,
        color="red",
        linewidths=2.2,
        label="Lagrangian points",
        zorder=10,
    )

    # Labels.
    x_label_offset = 0.02 * (xlim[1] - xlim[0])

    # ax.text(x_label_offset, L_outer_donor, r"$L_{\rm out,donor}$", va="center")
    # ax.text(x_label_offset, L1, r"$L_1$", va="center")
    # ax.text(x_label_offset, L_outer_accretor, r"$L_{\rm out,accretor}$", va="center")
    # ax.text(x_label_offset, 0.0, "donor", va="center")
    # ax.text(x_label_offset, 1.0, "accretor", va="center")
    # ax.text(x_label_offset, yG, "G", va="center")

    ax.set_xlabel(r"$x/a$")
    ax.set_ylabel(r"$y/a$")
    ax.set_title(
        rf"Roche trajectories with radiation pressure: "
        rf"$q={q:.3g}$, $\Gamma={Gamma:.3g}$"
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    info_text = (
        rf"$\mu_1={mu1:.3g}$" "\n"
        rf"$\mu_2={mu2:.3g}$" "\n"
        rf"$\mu_{{1,\rm eff}}=(1-\Gamma)\mu_1={mu1_eff:.3g}$" "\n"
        rf"$y_G={yG:.3g}$" "\n"
        rf"$\Phi(L_{{\rm out,donor}})={critical_phi_values[0]:.3g}$" "\n"
        rf"$\Phi(L_1)={critical_phi_values[1]:.3g}$" "\n"
        rf"$\Phi(L_{{\rm out,accretor}})={critical_phi_values[2]:.3g}$"
    )

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_file, dpi=250)
    plt.show()


def parse_limits(values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None

    if len(values) != 2:
        raise ValueError("Limits must contain exactly two numbers.")

    return float(values[0]), float(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Roche trajectories over a segmented modified Roche potential. "
            "The segmented colors and isocontours use the same potential levels, "
            "including the three Lagrangian point potentials."
        )
    )

    parser.add_argument(
        "summary_file",
        nargs="?",
        default="roche_summary.csv",
        help="Summary CSV file produced by the C++ code.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="roche_trajectories_segmented.png",
        help="Output image file.",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=700,
        help="Grid size for the Roche potential background.",
    )

    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Manual x limits.",
    )

    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Manual y limits.",
    )

    parser.add_argument(
        "--softening",
        type=float,
        default=1.0e-3,
        help="Small softening length used only for plotting the singular potential.",
    )

    parser.add_argument(
        "--potential-segments",
        type=int,
        default=10,
        help=(
            "Number of color bands for the modified Roche potential. "
            "Default is 10. The three Lagrangian point potentials are "
            "included as segmentation boundaries."
        ),
    )

    parser.add_argument(
        "--cmap",
        default="gist_earth",
        help="Matplotlib colormap name. Default is gist_earth.",
    )

    args = parser.parse_args()

    plot_roche_trajectories(
        summary_file=Path(args.summary_file),
        output_file=Path(args.output),
        grid_size=args.grid_size,
        xlim=parse_limits(args.xlim),
        ylim=parse_limits(args.ylim),
        potential_softening=args.softening,
        potential_segments=args.potential_segments,
        colormap=args.cmap,
    )


if __name__ == "__main__":
    main()