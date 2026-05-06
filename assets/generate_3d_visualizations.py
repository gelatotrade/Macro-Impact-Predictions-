"""
3D Visualization Generator for Macro Event Impact Prediction System

Each animation keeps a FIXED camera view. Instead of rotating the scene,
the underlying model variables are animated and the surface/heights
morph in response. A text overlay on each frame shows the current
variable values driving the surface.

Outputs:
1. market_implied_predictions  - VIX sweeps; surface lifts as VIX rises
2. multi_asset_coverage        - Event impact cycles; bars react per asset
3. scenario_analysis           - Event category flips inflation vs growth
4. probability_distribution    - Mean (bias) and σ (VIX) animate
5. real_time_analysis          - Time advances; live tick scrolls in
6. economic_calendar           - "Today" cursor advances; nearest events grow
7. risk_assessment             - VIX rises; risk peak sharpens & climbs
8. yield_curve                 - Fed front-end and inflation back-end shift
9. interactive_dashboard       - All dashboard indicators stream new values
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUTPUT_DIR = Path(__file__).parent / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bump this suffix when you change the GIF rendering — it busts the
# CDN/browser cache because the README image URLs change with it.
GIF_SUFFIX = "_v2"

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
FRAMES = 80
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


def _hud(ax, lines, x=0.02, y=0.98):
    """Draw a heads-up overlay on the 2D figure (above 3D axes)."""
    text = "\n".join(lines)
    ax.text2D(
        x, y, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, color="#ffffff",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#0d0d1a", edgecolor="#2196F3", alpha=0.85),
    )


def _save(fig, anim, name):
    out = OUTPUT_DIR / f"{name}{GIF_SUFFIX}.gif"
    anim.save(str(out), writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"  saved {out.name}")


def _ramp(frame, lo, hi):
    """Triangle wave 0..1..0 over FRAMES."""
    t = frame / (FRAMES - 1)
    tri = 1 - abs(2 * t - 1)
    return lo + (hi - lo) * tri


# ---------------------------------------------------------------------
# 1. Market-Implied Predictions
# ---------------------------------------------------------------------
def market_implied_predictions():
    """Surface: Expected SPY Move = f(VIX, P(Fed Cut)).
    VIX sweeps 12 → 35 → 12. The whole surface rises as VIX rises."""
    print("market_implied_predictions...")
    vix_axis = np.linspace(10, 40, 50)
    fed_axis = np.linspace(0, 1, 50)
    V, F = np.meshgrid(vix_axis, fed_axis)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Market-Implied Expected SPY Move (%)")
        # Animated variable: realized VIX point on surface
        vix_now = _ramp(frame, 12.0, 35.0)
        fed_now = 0.35 + 0.25 * np.sin(frame / FRAMES * 2 * np.pi)
        # The model: daily move = VIX/sqrt(252), scaled by Fed-cut bias
        Z = (V / np.sqrt(252)) * (1 + 0.6 * F)
        # Implied-vol scaling factor that lifts/lowers the whole surface
        vol_factor = vix_now / 18.0
        Z = Z * vol_factor
        ax.plot_surface(V, F, Z, cmap=cm.plasma, alpha=0.85,
                        linewidth=0, antialiased=True)
        # Mark the current (VIX, P(Cut)) point on the surface
        z_now = (vix_now / np.sqrt(252)) * (1 + 0.6 * fed_now) * vol_factor
        ax.scatter([vix_now], [fed_now], [z_now],
                   color="#00C851", s=80, edgecolor="white", linewidth=1.5)
        ax.plot([vix_now, vix_now], [fed_now, fed_now], [0, z_now],
                color="#00C851", linewidth=1.2, linestyle="--")
        ax.set_xlabel("VIX")
        ax.set_ylabel("P(Fed Cut)")
        ax.set_zlabel("Expected Move %")
        ax.set_xlim(10, 40)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 5)
        ax.view_init(elev=25, azim=-60)
        _hud(ax, [
            f"VIX        = {vix_now:5.2f}",
            f"P(Fed Cut) = {fed_now*100:4.1f}%",
            f"Expected   = {z_now:4.2f}%",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "market_implied_predictions")


# ---------------------------------------------------------------------
# 2. Multi-Asset Coverage
# ---------------------------------------------------------------------
def multi_asset_coverage():
    """Bars per instrument react as the event impact level cycles
    LOW → MEDIUM → HIGH → CRITICAL → ..."""
    print("multi_asset_coverage...")
    instruments = ["SPY", "QQQ", "IWM", "TLT", "IEF", "DXY", "EUR", "JPY", "Gold"]
    asset_class = ["EQ", "EQ", "EQ", "FI", "FI", "FX", "FX", "FX", "CM"]
    type_mult = {"EQ": 1.0, "FI": 0.7, "FX": 0.5, "CM": 0.8}
    colors = {"EQ": "#4CAF50", "FI": "#2196F3", "FX": "#FF9800", "CM": "#9C27B0"}
    base = np.array([type_mult[c] for c in asset_class])

    impacts = [
        ("LOW",      0.5),
        ("MEDIUM",   0.8),
        ("HIGH",     1.2),
        ("CRITICAL", 1.8),
    ]
    xs = np.arange(len(instruments))
    bar_colors = [colors[c] for c in asset_class]

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Multi-Asset Expected Moves (1σ %)")
        # Variable: cycle through impact levels, smooth interpolation
        pos = (frame / FRAMES) * len(impacts)
        i0 = int(pos) % len(impacts)
        i1 = (i0 + 1) % len(impacts)
        frac = pos - int(pos)
        label = impacts[i0][0] if frac < 0.5 else impacts[i1][0]
        mult = impacts[i0][1] * (1 - frac) + impacts[i1][1] * frac
        # VIX baseline 1.1% daily SPX move
        daily = 1.1
        zs = base * daily * mult
        ax.bar3d(xs, np.zeros_like(xs), np.zeros_like(xs),
                 0.6, 0.6, zs,
                 color=bar_colors, alpha=0.9, shade=True)
        for x, z in zip(xs, zs):
            ax.text(x + 0.3, 0.3, z + 0.05, f"{z:.2f}%",
                    color="white", fontsize=8, ha="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(instruments, rotation=30, ha="right")
        ax.set_yticks([])
        ax.set_zlim(0, 2.5)
        ax.set_zlabel("Expected Move %")
        ax.view_init(elev=25, azim=-55)
        _hud(ax, [
            f"Event Impact = {label}",
            f"Multiplier   = {mult:.2f}x",
            "Daily SPX σ  = 1.10%",
            "EQ green | FI blue | FX orange | CM purple",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "multi_asset_coverage")


# ---------------------------------------------------------------------
# 3. Scenario Analysis
# ---------------------------------------------------------------------
def scenario_analysis():
    """The same scenario grid flips signs for inflation vs growth events.
    Cycle the event category."""
    print("scenario_analysis...")
    scenarios = ["Large Miss", "Miss", "Inline", "Beat", "Large Beat"]
    instruments = ["SPY", "QQQ", "TLT", "DXY", "Gold"]

    # For growth events: beat = up for risk assets
    growth = np.array([
        [-1.8, -2.0, +1.5, -0.6, +0.8],   # Large Miss
        [-0.9, -1.0, +0.7, -0.3, +0.4],   # Miss
        [+0.1, +0.1, +0.0, +0.0, +0.0],   # Inline
        [+0.9, +1.0, -0.7, +0.3, -0.4],   # Beat
        [+1.8, +2.0, -1.5, +0.6, -0.8],   # Large Beat
    ])
    # For inflation events: beat = down for risk assets
    inflation = np.array([
        [+1.8, +2.0, -1.5, -0.8, +1.2],   # Large Miss
        [+0.9, +1.0, -0.7, -0.4, +0.6],
        [+0.1, +0.1, +0.0, +0.0, +0.0],
        [-0.9, -1.0, +0.7, +0.4, -0.6],
        [-1.8, -2.0, +1.5, +0.8, -1.2],
    ])
    categories = [("Growth (NFP/GDP)", growth),
                  ("Inflation (CPI/PCE)", inflation)]

    X, Y = np.meshgrid(np.arange(len(instruments)), np.arange(len(scenarios)))

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Scenario Analysis: Move % by Outcome")
        pos = (frame / FRAMES) * len(categories) * 2
        idx = int(pos) % len(categories)
        next_idx = (idx + 1) % len(categories)
        frac = pos - int(pos)
        # Smooth cross-fade between two category surfaces
        ease = 0.5 - 0.5 * np.cos(frac * np.pi)
        Z = categories[idx][1] * (1 - ease) + categories[next_idx][1] * ease
        label = categories[idx][0] if ease < 0.5 else categories[next_idx][0]
        ax.plot_surface(X, Y, Z, cmap=cm.RdYlGn, alpha=0.9,
                        linewidth=0.4, edgecolor="#1a1a2e",
                        vmin=-2.5, vmax=2.5)
        ax.set_xticks(np.arange(len(instruments)))
        ax.set_xticklabels(instruments)
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_yticklabels(scenarios, fontsize=8)
        ax.set_zlabel("Move %")
        ax.set_zlim(-2.5, 2.5)
        ax.view_init(elev=25, azim=-50)
        _hud(ax, [
            f"Event Category = {label}",
            "Beat → equities " + ("UP" if "Growth" in label else "DOWN"),
            "Miss → equities " + ("DOWN" if "Growth" in label else "UP"),
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "scenario_analysis")


# ---------------------------------------------------------------------
# 4. Probability Distribution
# ---------------------------------------------------------------------
def probability_distribution():
    """Density surface (move% × σ × density). Both the mean (directional
    bias) and σ (VIX) animate."""
    print("probability_distribution...")
    x = np.linspace(-4, 4, 70)
    s = np.linspace(0.4, 2.0, 70)
    X, S = np.meshgrid(x, s)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Expected-Move Probability Density")
        phase = frame / FRAMES * 2 * np.pi
        mean_shift = 0.8 * np.sin(phase)              # directional bias
        sigma_scale = 1.0 + 0.5 * np.sin(phase * 0.5)  # vol regime
        S_eff = S * sigma_scale
        Z = (1 / (S_eff * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((X - mean_shift) / S_eff) ** 2)
        ax.plot_surface(X, S, Z, cmap=cm.viridis, alpha=0.85,
                        linewidth=0, antialiased=True, vmin=0, vmax=1)
        ax.set_xlabel("Move %")
        ax.set_ylabel("σ axis")
        ax.set_zlabel("Density")
        ax.set_zlim(0, 1.1)
        ax.view_init(elev=30, azim=-55)
        bias_dir = "BULLISH" if mean_shift > 0.05 else \
                   "BEARISH" if mean_shift < -0.05 else "NEUTRAL"
        _hud(ax, [
            f"Mean μ      = {mean_shift:+.2f}%   ({bias_dir})",
            f"σ scale     = {sigma_scale:.2f}",
            f"P(Up)       ≈ {(0.5 + mean_shift/4)*100:4.1f}%",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "probability_distribution")


# ---------------------------------------------------------------------
# 5. Real-Time Analysis
# ---------------------------------------------------------------------
def real_time_analysis():
    """A scrolling time-series wave: time advances, the latest tick's
    VIX/Fed/Inflation values are written to the HUD."""
    print("real_time_analysis...")
    rng = np.random.default_rng(11)
    t = np.linspace(0, 6 * np.pi, 80)
    inst = np.linspace(0, 1, 30)
    T, I = np.meshgrid(t, inst)
    # Pre-generated tick history
    vix_series = 18 + 4 * np.sin(np.linspace(0, 4 * np.pi, FRAMES)) + \
        rng.normal(0, 0.4, FRAMES)
    fed_series = 0.35 + 0.15 * np.cos(np.linspace(0, 3 * np.pi, FRAMES))
    infl_series = 2.5 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, FRAMES))

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Real-Time Expected-Move Stream")
        phase = frame / FRAMES * 2 * np.pi
        # Wave speed scales with current VIX (more vol = faster wiggles)
        vol_factor = vix_series[frame] / 18.0
        Z = np.sin(T - phase) * np.cos(I * np.pi - phase * 0.3) * \
            (1 + 0.4 * I) * vol_factor
        ax.plot_surface(T, I, Z, cmap=cm.coolwarm, alpha=0.9,
                        linewidth=0, antialiased=True, vmin=-2, vmax=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Instrument")
        ax.set_zlabel("Move %")
        ax.set_zlim(-2.5, 2.5)
        ax.view_init(elev=22, azim=-55)
        _hud(ax, [
            f"Tick           = t+{frame:02d}s",
            f"VIX            = {vix_series[frame]:5.2f}",
            f"P(Fed Cut)     = {fed_series[frame]*100:4.1f}%",
            f"Breakeven Infl = {infl_series[frame]:.2f}%",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "real_time_analysis")


# ---------------------------------------------------------------------
# 6. Economic Calendar
# ---------------------------------------------------------------------
def economic_calendar():
    """A 'today' cursor advances day by day. The bar at today's index
    pulses; events within 2 days highlight."""
    print("economic_calendar...")
    rng = np.random.default_rng(7)
    n = 14
    days = np.arange(n)
    impact = rng.choice([1, 2, 3, 4], size=n, p=[0.3, 0.3, 0.25, 0.15])
    height = impact.astype(float) + rng.uniform(0, 0.5, n)
    cmap = {1: "#9E9E9E", 2: "#ffbb33", 3: "#FF9800", 4: "#ff4444"}
    base_colors = [cmap[i] for i in impact]
    events = ["CPI", "NFP", "FOMC", "PMI", "GDP", "PPI", "ADP",
              "Claims", "Retail", "ISM", "Sales", "Core PCE", "Housing", "Sentiment"]

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Economic Calendar - Upcoming High-Impact Events")
        today = (frame / FRAMES) * (n - 1)
        # Bars within 2 days of "today" are emphasized
        dist = np.abs(days - today)
        boost = np.clip(1.5 - dist * 0.4, 0.6, 1.8)
        h = height * boost
        # Pulse the bar nearest "today"
        nearest = int(round(today))
        pulse = 1 + 0.2 * np.sin(frame / FRAMES * 4 * np.pi)
        h = h.copy()
        h[nearest] *= pulse
        ax.bar3d(days, np.zeros(n), np.zeros(n),
                 0.7, 0.7, h, color=base_colors, alpha=0.9, shade=True)
        # "today" line
        ax.plot([today, today], [0, 1], [0, 0],
                color="#00E5FF", linewidth=2)
        ax.text(today, 0.5, h[nearest] + 0.4, "TODAY",
                color="#00E5FF", fontsize=9, ha="center")
        ax.set_xticks(days)
        ax.set_xticklabels(events, rotation=45, ha="right", fontsize=7)
        ax.set_yticks([])
        ax.set_zlim(0, 6)
        ax.set_zlabel("Impact Level")
        ax.view_init(elev=28, azim=-65)
        next_evt = events[min(nearest + 1, n - 1)]
        next_imp = ["LOW", "MED", "HIGH", "CRIT"][impact[min(nearest + 1, n - 1)] - 1]
        _hud(ax, [
            f"Day        = {nearest+1}/{n}",
            f"Today      = {events[nearest]}  ({['LOW','MED','HIGH','CRIT'][impact[nearest]-1]})",
            f"Next event = {next_evt} ({next_imp})",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "economic_calendar")


# ---------------------------------------------------------------------
# 7. Risk Assessment
# ---------------------------------------------------------------------
def risk_assessment():
    """Risk = combined VIX × event impact. As VIX rises the risk peak
    sharpens and lifts. Camera fixed."""
    print("risk_assessment...")
    x = np.linspace(-3, 3, 60)
    y = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Event Risk Surface")
        vix_now = _ramp(frame, 12.0, 35.0)
        impact = ["LOW", "MEDIUM", "HIGH", "CRITICAL"][min(3, int(vix_now / 8))]
        # Sharpness grows with VIX
        sharp = 0.6 + 0.04 * vix_now
        peak = vix_now / 5.0
        R = np.sqrt(X**2 + Y**2)
        Z = peak * np.exp(-(R * sharp / 3) ** 2)
        ax.plot_surface(X, Y, Z, cmap=cm.inferno, alpha=0.9,
                        linewidth=0, antialiased=True, vmin=0, vmax=10)
        ax.set_xlabel("VIX delta")
        ax.set_ylabel("Impact delta")
        ax.set_zlabel("Risk Score")
        ax.set_zlim(0, 12)
        ax.view_init(elev=30, azim=-55)
        risk_score = round(min((vix_now / 20 + 0.6 * (sharp - 0.6) * 4) * 3, 10), 1)
        risk_level = "LOW" if risk_score < 3 else \
                     "MEDIUM" if risk_score < 6 else \
                     "HIGH" if risk_score < 8 else "EXTREME"
        _hud(ax, [
            f"VIX          = {vix_now:5.2f}",
            f"Event Impact = {impact}",
            f"Risk Score   = {risk_score:4.1f} / 10",
            f"Risk Level   = {risk_level}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "risk_assessment")


# ---------------------------------------------------------------------
# 8. Yield Curve
# ---------------------------------------------------------------------
def yield_curve():
    """Yield curve evolves over a rolling time window. The Fed front-end
    and the inflation back-end animate independently. Camera fixed."""
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
        # Two independent variables: front-end (Fed) and back-end
        front = 4.5 + 0.8 * np.sin(phase)
        back = 4.0 + 0.5 * np.cos(phase * 0.7)
        slope = (back - front) / 30
        Z = front + slope * M + 0.10 * np.sin(T * 0.3 + phase)
        ax.plot_surface(T, M, Z, cmap=cm.cividis, alpha=0.9,
                        linewidth=0, antialiased=True, vmin=2.5, vmax=5.5)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Tenor (years)")
        ax.set_zlabel("Yield %")
        ax.set_zlim(2.5, 5.5)
        ax.view_init(elev=25, azim=-55)
        spread_bp = (back - front) * 100
        shape = "INVERTED" if spread_bp < -10 else \
                "FLAT" if abs(spread_bp) < 10 else \
                "NORMAL"
        _hud(ax, [
            f"Front (3M) = {front:.2f}%",
            f"Long  (30Y) = {back:.2f}%",
            f"Slope       = {spread_bp:+5.0f} bp",
            f"Shape       = {shape}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "yield_curve")


# ---------------------------------------------------------------------
# 9. Interactive Dashboard
# ---------------------------------------------------------------------
def interactive_dashboard():
    """Six dashboard indicators as towers. Each tower height streams
    new values; the HUD lists the current numeric reading."""
    print("interactive_dashboard...")
    indicators = ["VIX", "P(FedCut)%", "10Y-3M", "Infl bp",
                  "Risk", "Daily σ"]
    base = np.array([18.0, 35.0, 0.40, 2.50, 5.0, 1.10])
    swing = np.array([4.0, 12.0, 0.20, 0.30, 1.5, 0.25])
    n = len(indicators)
    grid_x = (np.arange(n) % 3).astype(float)
    grid_y = (np.arange(n) // 3).astype(float)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Interactive Dashboard - Live Indicators")
        phase = frame / FRAMES * 2 * np.pi
        vals = base + swing * np.sin(phase + np.arange(n) * 0.7)
        # Normalize each indicator into 0..10 visual scale by its swing
        norm_h = ((vals - (base - swing)) / (2 * swing)) * 10
        colors = cm.plasma(norm_h / 10)
        ax.bar3d(grid_x, grid_y, np.zeros(n),
                 0.6, 0.6, norm_h, color=colors, alpha=0.9, shade=True)
        for i, name in enumerate(indicators):
            ax.text(grid_x[i] + 0.3, grid_y[i] + 0.3, norm_h[i] + 0.4,
                    name, color="white", fontsize=8, ha="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zlim(0, 12)
        ax.set_zlabel("Normalized")
        ax.view_init(elev=30, azim=-55)
        _hud(ax, [
            f"VIX         = {vals[0]:5.2f}",
            f"P(Fed Cut)  = {vals[1]:4.1f}%",
            f"10Y-3M      = {vals[2]:+.2f}",
            f"Breakeven   = {vals[3]:.2f}%",
            f"Risk        = {vals[4]:.1f}/10",
            f"Daily σ     = {vals[5]:.2f}%",
        ])
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
