"""
3D Visualization Generator for Macro Event Impact Prediction System

Generates animated 3D GIFs for each major feature of the system:
1. Market-Implied Predictions   - 3D VIX/Yield/Fed surface
2. Multi-Asset Coverage         - 3D bars rotating across asset classes
3. Scenario Analysis            - 3D scenario landscape
4. Probability Distributions    - 3D probability density surface
5. Real-Time Analysis           - 3D time-series wave
6. Economic Calendar            - 3D event timeline
7. Risk Assessment              - 3D risk gauge with sweep
8. Yield Curve Animation        - 3D yield curve evolution

Each GIF is saved to assets/visualizations/<feature>.gif
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUTPUT_DIR = Path(__file__).parent / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visual style settings
plt.rcParams.update({
    "axes.facecolor": "#1a1a2e",
    "figure.facecolor": "#1a1a2e",
    "axes.edgecolor": "#2d2d44",
    "axes.labelcolor": "#ffffff",
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
    "text.color": "#ffffff",
    "axes.titlecolor": "#ffffff",
    "grid.color": "#2d2d44",
})

FPS = 20
FRAMES = 60
DPI = 80
SIZE = (8, 6)


def _setup_3d(ax, title):
    ax.set_facecolor("#1a1a2e")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#2d2d44")
    ax.yaxis.pane.set_edgecolor("#2d2d44")
    ax.zaxis.pane.set_edgecolor("#2d2d44")
    ax.set_title(title, fontsize=13, color="white", pad=14)


def _save(fig, anim, name):
    out = OUTPUT_DIR / f"{name}.gif"
    anim.save(str(out), writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"  saved {out.name}")


def market_implied_predictions():
    """3D surface showing how VIX, Fed-cut probability and yield-spread
    interact to drive expected SPY moves."""
    print("market_implied_predictions...")
    vix = np.linspace(10, 40, 40)
    fed = np.linspace(0, 1, 40)
    V, F = np.meshgrid(vix, fed)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")
    _setup_3d(ax, "Market-Implied Expected SPY Move (%)")
    ax.set_xlabel("VIX")
    ax.set_ylabel("P(Fed Cut)")
    ax.set_zlabel("Expected Move %")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Market-Implied Expected SPY Move (%)")
        ax.set_xlabel("VIX")
        ax.set_ylabel("P(Fed Cut)")
        ax.set_zlabel("Expected Move %")
        phase = frame / FRAMES * 2 * np.pi
        Z = (V / np.sqrt(252)) * (1 + 0.6 * F) * (1 + 0.15 * np.sin(phase))
        ax.plot_surface(V, F, Z, cmap=cm.plasma, alpha=0.85,
                        linewidth=0, antialiased=True)
        ax.view_init(elev=25, azim=frame * 6)
        ax.set_zlim(0, 4)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "market_implied_predictions")


def multi_asset_coverage():
    """3D bar chart of expected moves across instrument classes."""
    print("multi_asset_coverage...")
    instruments = ["SPY", "QQQ", "IWM", "TLT", "IEF", "DXY", "EUR", "JPY", "Gold"]
    asset_class = ["EQ", "EQ", "EQ", "FI", "FI", "FX", "FX", "FX", "CM"]
    colors = {"EQ": "#4CAF50", "FI": "#2196F3", "FX": "#FF9800", "CM": "#9C27B0"}
    base_values = np.array([0.95, 1.10, 1.30, 0.80, 0.60, 0.45, 0.55, 0.65, 0.70])
    rng = np.random.default_rng(42)
    waves = rng.uniform(0.10, 0.35, size=len(instruments))

    xs = np.arange(len(instruments))
    ys = np.zeros_like(xs)
    bar_colors = [colors[c] for c in asset_class]

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Multi-Asset Expected Moves (1σ %)")
        phase = frame / FRAMES * 2 * np.pi
        zs = base_values + waves * np.sin(phase + xs * 0.4)
        ax.bar3d(xs, ys, np.zeros_like(xs), 0.6, 0.6, zs,
                 color=bar_colors, alpha=0.9, shade=True)
        ax.set_xticks(xs)
        ax.set_xticklabels(instruments, rotation=30, ha="right")
        ax.set_yticks([])
        ax.set_zlim(0, 2.0)
        ax.set_zlabel("Expected Move %")
        ax.view_init(elev=25, azim=-60 + frame * 4)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "multi_asset_coverage")


def scenario_analysis():
    """3D ribbon showing how different surprise scenarios fan out
    across instruments."""
    print("scenario_analysis...")
    scenarios = ["Large Miss", "Miss", "Inline", "Beat", "Large Beat"]
    instruments = ["SPY", "QQQ", "TLT", "DXY", "Gold"]
    base = np.array([
        [+1.8, +2.0, -1.5, -0.8, +1.2],   # Large Miss
        [+0.9, +1.0, -0.7, -0.4, +0.6],   # Miss
        [+0.1, +0.1, +0.0, +0.0, +0.0],   # Inline
        [-0.9, -1.0, +0.7, +0.4, -0.6],   # Beat
        [-1.8, -2.0, +1.5, +0.8, -1.2],   # Large Beat (inflation event)
    ])
    X, Y = np.meshgrid(np.arange(len(instruments)), np.arange(len(scenarios)))

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Scenario Analysis: Move % by Outcome")
        wobble = 1 + 0.08 * np.sin(frame / FRAMES * 2 * np.pi)
        Z = base * wobble
        ax.plot_surface(X, Y, Z, cmap=cm.RdYlGn, alpha=0.9,
                        linewidth=0.4, edgecolor="#1a1a2e")
        ax.set_xticks(np.arange(len(instruments)))
        ax.set_xticklabels(instruments)
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_yticklabels(scenarios, fontsize=8)
        ax.set_zlabel("Move %")
        ax.set_zlim(-2.5, 2.5)
        ax.view_init(elev=25, azim=-45 + frame * 6)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "scenario_analysis")


def probability_distribution():
    """3D probability density surface (mean × stddev × density)."""
    print("probability_distribution...")
    x = np.linspace(-4, 4, 60)
    s = np.linspace(0.4, 2.0, 60)
    X, S = np.meshgrid(x, s)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Expected-Move Probability Density")
        mean_shift = 0.8 * np.sin(frame / FRAMES * 2 * np.pi)
        Z = (1 / (S * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mean_shift) / S) ** 2)
        ax.plot_surface(X, S, Z, cmap=cm.viridis, alpha=0.85,
                        linewidth=0, antialiased=True)
        ax.set_xlabel("Move %")
        ax.set_ylabel("σ (vol)")
        ax.set_zlabel("Density")
        ax.set_zlim(0, 1.1)
        ax.view_init(elev=30, azim=frame * 6)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "probability_distribution")


def real_time_analysis():
    """3D rolling wave of expected moves over time as new data arrives."""
    print("real_time_analysis...")
    t = np.linspace(0, 6 * np.pi, 80)
    inst = np.linspace(0, 1, 30)
    T, I = np.meshgrid(t, inst)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Real-Time Expected-Move Stream")
        phase = frame / FRAMES * 2 * np.pi
        Z = np.sin(T - phase) * np.cos(I * np.pi - phase * 0.3) * (1 + 0.4 * I)
        ax.plot_surface(T, I, Z, cmap=cm.coolwarm, alpha=0.9,
                        linewidth=0, antialiased=True)
        ax.set_xlabel("Time")
        ax.set_ylabel("Instrument")
        ax.set_zlabel("Move %")
        ax.set_zlim(-2, 2)
        ax.view_init(elev=22, azim=-60 + frame * 5)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "real_time_analysis")


def economic_calendar():
    """3D timeline of upcoming events colored by impact level."""
    print("economic_calendar...")
    rng = np.random.default_rng(7)
    n = 14
    days = np.arange(n)
    impact = rng.choice([1, 2, 3, 4], size=n, p=[0.3, 0.3, 0.25, 0.15])
    height = impact.astype(float) + rng.uniform(0, 0.5, n)
    cmap = {1: "#9E9E9E", 2: "#ffbb33", 3: "#FF9800", 4: "#ff4444"}
    bar_colors = [cmap[i] for i in impact]
    events = ["CPI", "NFP", "FOMC", "PMI", "GDP", "PPI", "ADP",
              "Claims", "Retail", "ISM", "Sales", "Core PCE", "Housing", "Sentiment"]

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Economic Calendar - Upcoming High-Impact Events")
        pulse = 1 + 0.12 * np.sin(frame / FRAMES * 2 * np.pi + days * 0.4)
        ax.bar3d(days, np.zeros(n), np.zeros(n),
                 0.7, 0.7, height * pulse,
                 color=bar_colors, alpha=0.9, shade=True)
        ax.set_xticks(days)
        ax.set_xticklabels(events, rotation=45, ha="right", fontsize=7)
        ax.set_yticks([])
        ax.set_zlim(0, 5)
        ax.set_zlabel("Impact Level")
        ax.view_init(elev=28, azim=-70 + frame * 4)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "economic_calendar")


def risk_assessment():
    """3D pyramid-style risk surface with rotating sweep."""
    print("risk_assessment...")
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Event Risk Surface (VIX × Impact)")
        phase = frame / FRAMES * 2 * np.pi
        Z = 10 * np.exp(-R) * (1 + 0.25 * np.sin(phase + R))
        ax.plot_surface(X, Y, Z, cmap=cm.inferno, alpha=0.9,
                        linewidth=0, antialiased=True)
        ax.set_xlabel("VIX delta")
        ax.set_ylabel("Impact delta")
        ax.set_zlabel("Risk Score")
        ax.set_zlim(0, 12)
        ax.view_init(elev=30, azim=frame * 6)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "risk_assessment")


def yield_curve():
    """Animated 3D yield-curve evolution over rolling time window."""
    print("yield_curve...")
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    times = np.arange(40)
    T, M = np.meshgrid(times, tenors)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Yield Curve Evolution")
        phase = frame / FRAMES * 2 * np.pi
        # Curve shape: short-end driven by Fed, long-end by growth/inflation
        front = 4.5 + 0.6 * np.sin(phase)
        back = 4.0 + 0.4 * np.cos(phase * 0.7)
        slope = (back - front) / 30
        Z = front + slope * M + 0.15 * np.sin(T * 0.3 + phase)
        ax.plot_surface(T, M, Z, cmap=cm.cividis, alpha=0.9,
                        linewidth=0, antialiased=True)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Tenor (years)")
        ax.set_zlabel("Yield %")
        ax.set_zlim(2.5, 5.5)
        ax.view_init(elev=25, azim=-50 + frame * 5)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "yield_curve")


def interactive_dashboard():
    """3D landscape with multiple bar towers representing dashboard
    indicators (VIX, Fed, Spread, Inflation, Regime)."""
    print("interactive_dashboard...")
    indicators = ["VIX", "Fed Cut%", "Curve", "Inflation", "Regime", "Daily σ"]
    base = np.array([18, 35, 0.4, 2.5, 5.0, 1.1])
    scale = np.array([3.0, 8.0, 0.15, 0.3, 1.5, 0.25])

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Interactive Dashboard - Live Indicators")
        phase = frame / FRAMES * 2 * np.pi
        n = len(indicators)
        xs = np.arange(n)
        # Build a 3x2 grid of indicator towers
        grid_x = (xs % 3).astype(float)
        grid_y = (xs // 3).astype(float)
        heights = base + scale * np.sin(phase + xs * 0.7)
        # Normalize to similar visual scale (0..10)
        norm_h = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6) * 10 + 1
        colors = cm.plasma(norm_h / 11)
        ax.bar3d(grid_x, grid_y, np.zeros(n),
                 0.6, 0.6, norm_h, color=colors, alpha=0.9, shade=True)
        for i, name in enumerate(indicators):
            ax.text(grid_x[i] + 0.3, grid_y[i] + 0.3, norm_h[i] + 0.5,
                    f"{name}\n{heights[i]:.2f}",
                    color="white", fontsize=8, ha="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zlim(0, 13)
        ax.set_zlabel("Normalized")
        ax.view_init(elev=30, azim=-70 + frame * 5)
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "interactive_dashboard")


def main():
    print(f"Generating GIFs into {OUTPUT_DIR}\n")
    market_implied_predictions()
    multi_asset_coverage()
    scenario_analysis()
    probability_distribution()
    real_time_analysis()
    economic_calendar()
    risk_assessment()
    yield_curve()
    interactive_dashboard()
    print("\nDone.")


if __name__ == "__main__":
    main()
