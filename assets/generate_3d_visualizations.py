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


# ---------------------------------------------------------------------
# 10. Fed Funds Futures
# ---------------------------------------------------------------------
def fed_funds_futures():
    """Stacked bars per upcoming FOMC meeting. Animate: time progresses,
    P(Cut)/P(Hold)/P(Hike) updates."""
    print("fed_funds_futures...")
    meetings = ["Mar", "May", "Jun", "Jul", "Sep", "Nov"]
    n = len(meetings)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Fed Funds Futures - Implied FOMC Path")
        # Animated variable: market mood shifts dovish→hawkish→dovish
        mood = np.sin(frame / FRAMES * 2 * np.pi)  # +1 = dovish, -1 = hawkish
        idx = np.arange(n)
        # Probabilities evolve with horizon: closer meetings have sharper conviction
        horizon_decay = np.exp(-idx * 0.3)
        p_cut = np.clip(0.35 + 0.4 * mood * horizon_decay, 0.0, 0.95)
        p_hike = np.clip(0.15 - 0.3 * mood * horizon_decay, 0.0, 0.95)
        p_hold = np.clip(1 - p_cut - p_hike, 0.0, 1.0)
        # Stacked bars
        ax.bar3d(idx, np.zeros(n), np.zeros(n),
                 0.6, 0.6, p_cut * 100,
                 color="#00C851", alpha=0.9, shade=True)
        ax.bar3d(idx, np.zeros(n), p_cut * 100,
                 0.6, 0.6, p_hold * 100,
                 color="#ffbb33", alpha=0.9, shade=True)
        ax.bar3d(idx, np.zeros(n), (p_cut + p_hold) * 100,
                 0.6, 0.6, p_hike * 100,
                 color="#ff4444", alpha=0.9, shade=True)
        ax.set_xticks(idx)
        ax.set_xticklabels(meetings)
        ax.set_yticks([])
        ax.set_zlim(0, 110)
        ax.set_zlabel("Probability %")
        ax.view_init(elev=25, azim=-60)
        bias = "DOVISH" if mood > 0.2 else "HAWKISH" if mood < -0.2 else "NEUTRAL"
        _hud(ax, [
            f"Bias        = {bias}",
            f"Next: P(Cut)  = {p_cut[0]*100:4.1f}%",
            f"Next: P(Hold) = {p_hold[0]*100:4.1f}%",
            f"Next: P(Hike) = {p_hike[0]*100:4.1f}%",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "fed_funds_futures")


# ---------------------------------------------------------------------
# 11. TIPS Spreads
# ---------------------------------------------------------------------
def tips_spreads():
    """Breakeven inflation surface: tenor × time → breakeven %.
    Animate the breakeven shifting up/down."""
    print("tips_spreads...")
    tenors = np.array([2, 5, 7, 10, 20, 30])
    times = np.arange(40)
    T, M = np.meshgrid(times, tenors)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "TIPS Breakeven Inflation Surface")
        phase = frame / FRAMES * 2 * np.pi
        # Animated: breakeven inflation drifts 1.8% → 3.2% → 1.8%
        be_5y = 2.5 + 0.7 * np.sin(phase)
        be_30y = 2.3 + 0.4 * np.cos(phase * 0.6)
        # Term-structure: short-end more reactive than long-end
        slope = (be_30y - be_5y) / 25
        Z = be_5y + slope * (M - 5) + 0.05 * np.sin(T * 0.4 + phase)
        ax.plot_surface(T, M, Z, cmap=cm.YlOrRd, alpha=0.9,
                        linewidth=0, antialiased=True, vmin=1.0, vmax=4.0)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Tenor (years)")
        ax.set_zlabel("Breakeven %")
        ax.set_zlim(1.0, 4.0)
        ax.view_init(elev=25, azim=-55)
        regime = "RISING" if be_5y > 2.7 else "FALLING" if be_5y < 2.3 else "STABLE"
        _hud(ax, [
            f"5Y Breakeven  = {be_5y:.2f}%",
            f"30Y Breakeven = {be_30y:.2f}%",
            f"Slope         = {(be_30y-be_5y)*100:+5.0f} bp",
            f"Regime        = {regime}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "tips_spreads")


# ---------------------------------------------------------------------
# 12. VIX Term Structure
# ---------------------------------------------------------------------
def vix_term_structure():
    """VIX futures term structure: spot/1m/3m/6m. Animate spot VIX
    spiking and curve flipping from contango to backwardation."""
    print("vix_term_structure...")
    tenors = ["Spot", "1M", "3M", "6M", "9M"]
    n = len(tenors)
    xs = np.arange(n)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "VIX Futures Term Structure")
        spot = _ramp(frame, 12.0, 38.0)
        # Long-dated futures mean-revert toward 18; spot diverges
        far_anchor = 18.0
        decay = np.array([0, 0.3, 0.55, 0.75, 0.85])
        levels = spot * (1 - decay) + far_anchor * decay
        # Color by structure
        is_backward = spot > far_anchor + 2
        bar_colors = ["#ff4444" if is_backward else "#4CAF50"] * n
        ax.bar3d(xs, np.zeros(n), np.zeros(n),
                 0.6, 0.6, levels,
                 color=bar_colors, alpha=0.9, shade=True)
        for x, lv in zip(xs, levels):
            ax.text(x + 0.3, 0.3, lv + 0.5, f"{lv:.1f}",
                    color="white", fontsize=8, ha="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(tenors)
        ax.set_yticks([])
        ax.set_zlim(0, 45)
        ax.set_zlabel("VIX Level")
        ax.view_init(elev=25, azim=-55)
        regime = ("EXTREME" if spot > 35 else "HIGH" if spot > 25
                  else "NORMAL" if spot > 15 else "LOW")
        structure = "BACKWARDATION" if is_backward else "CONTANGO"
        _hud(ax, [
            f"Spot VIX  = {spot:5.2f}",
            f"6M Futures = {levels[3]:5.2f}",
            f"Regime     = {regime}",
            f"Structure  = {structure}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "vix_term_structure")


# ---------------------------------------------------------------------
# 13. Live Fed Funds Target Rate (DFEDTARU)
# ---------------------------------------------------------------------
def fed_funds_target_rate():
    """Step plot of FFTR over last 24 months. Animate cursor walking
    through the rate-hiking → rate-cutting cycle."""
    print("fed_funds_target_rate...")
    n_months = 24
    months = np.arange(n_months)
    # Synthesize a hike-then-cut cycle: 2.5% → 5.5% → 4.0%
    rate = np.concatenate([
        np.linspace(2.5, 5.5, 12),
        np.full(6, 5.5),
        np.linspace(5.5, 4.0, 6),
    ])

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Live Fed Funds Target Rate (DFEDTARU)")
        cursor = int((frame / FRAMES) * (n_months - 1))
        ys = np.zeros(n_months)
        # Bars; the bar at the cursor is highlighted
        colors = ["#2196F3"] * n_months
        colors[cursor] = "#00E5FF"
        ax.bar3d(months, ys, np.zeros(n_months),
                 0.7, 0.4, rate,
                 color=colors, alpha=0.9, shade=True)
        # "today" cursor line
        ax.plot([cursor, cursor], [0.5, 0.5], [0, rate[cursor] + 0.5],
                color="#00E5FF", linewidth=2)
        ax.set_xlabel("Month index")
        ax.set_yticks([])
        ax.set_zlim(0, 7)
        ax.set_zlabel("FFTR %")
        ax.view_init(elev=25, azim=-55)
        cycle = ("HIKING" if cursor < 12 else
                 "PEAK" if cursor < 18 else "CUTTING")
        _hud(ax, [
            f"Cursor    = month {cursor+1}/{n_months}",
            f"FFTR now  = {rate[cursor]:.2f}%",
            f"FFTR start = {rate[0]:.2f}%",
            f"Cycle     = {cycle}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "fed_funds_target_rate")


# ---------------------------------------------------------------------
# 14. Manual Consensus Override
# ---------------------------------------------------------------------
def consensus_override():
    """Three side-by-side bars per indicator: Static Fallback / FRED Proxy
    / User Override. Animate: override toggles on, bars realign."""
    print("consensus_override...")
    indicators = ["CPI MoM", "NFP", "Fed Funds", "Core PCE", "Unemp."]
    static_v   = np.array([0.30, 180,  5.25, 0.20, 4.20])
    fred_v     = np.array([0.18, 232,  4.50, 0.16, 4.05])
    override_v = np.array([0.25, 175,  4.50, 0.18, 4.10])
    # Normalize each indicator to ~0-10 scale per-indicator
    span = np.maximum(np.abs(static_v - fred_v) * 4, 0.1)

    n = len(indicators)
    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def _norm(v):
        return ((v - fred_v) / span) * 3 + 5

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Consensus Sources: Fallback vs FRED vs Override")
        # Variable: override fades in over second half of animation
        ovr_fade = max(0, (frame / FRAMES - 0.4)) / 0.6
        ovr_fade = min(ovr_fade, 1.0)
        for i, label in enumerate(indicators):
            # Static at y=0, FRED at y=1, Override at y=2
            ax.bar3d([i], [0], [0], 0.5, 0.5, _norm(static_v)[i],
                     color="#9E9E9E", alpha=0.9, shade=True)
            ax.bar3d([i], [1], [0], 0.5, 0.5, _norm(fred_v)[i],
                     color="#2196F3", alpha=0.9, shade=True)
            ax.bar3d([i], [2], [0], 0.5, 0.5,
                     _norm(override_v)[i] * ovr_fade,
                     color="#FFC107", alpha=0.9, shade=True)
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(indicators, fontsize=7, rotation=20)
        ax.set_yticks([0.25, 1.25, 2.25])
        ax.set_yticklabels(["Static", "FRED", "Override"], fontsize=8)
        ax.set_zlim(0, 12)
        ax.set_zlabel("Normalized")
        ax.view_init(elev=25, azim=-60)
        active = "Override ON" if ovr_fade > 0.5 else (
            "Override loading…" if ovr_fade > 0 else "Override OFF")
        _hud(ax, [
            f"Active source = {'OVERRIDE' if ovr_fade>0.5 else 'FRED'}",
            f"{active}",
            f"CPI MoM: static {static_v[0]:.2f} | "
            f"FRED {fred_v[0]:.2f} | OVR {override_v[0]:.2f}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "consensus_override")


# ---------------------------------------------------------------------
# 15. Historical Sensitivity Matrix
# ---------------------------------------------------------------------
def sensitivity_matrix():
    """3D heatmap-style bars: events × instruments → |sensitivity|.
    Animate one event row highlighted at a time."""
    print("sensitivity_matrix...")
    events = ["CPI MoM", "Core CPI", "NFP", "Unemp", "FOMC", "Core PCE"]
    instruments = ["SPY", "QQQ", "TLT", "DXY", "EUR", "GLD"]
    # From surprise_calculator.SENSITIVITY_MATRIX
    M = np.array([
        [-0.40, -0.55, -0.60,  0.25, -0.22, -0.35],   # CPI MoM
        [-0.45, -0.60, -0.65,  0.28, -0.24,  0.00],   # Core CPI
        [ 0.25,  0.30, -0.35,  0.30, -0.25,  0.00],   # NFP
        [-0.20, -0.25,  0.30, -0.20,  0.18,  0.00],   # Unemp
        [-0.60, -0.80, -0.90,  0.45, -0.40, -0.55],   # FOMC
        [-0.35, -0.45, -0.50,  0.22,  0.00,  0.00],   # PCE
    ])

    nE, nI = M.shape
    X, Y = np.meshgrid(np.arange(nI), np.arange(nE))

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Sensitivity Matrix |1σ Surprise → Move %|")
        active_row = int((frame / FRAMES) * nE) % nE
        # Bars; highlight active row by scaling brightness
        for r in range(nE):
            for c in range(nI):
                h = abs(M[r, c])
                color = cm.RdYlGn((M[r, c] + 1) / 2)
                alpha = 0.95 if r == active_row else 0.4
                ax.bar3d(c, r, 0, 0.7, 0.7, h,
                         color=color, alpha=alpha, shade=True)
        ax.set_xticks(np.arange(nI))
        ax.set_xticklabels(instruments)
        ax.set_yticks(np.arange(nE))
        ax.set_yticklabels(events, fontsize=8)
        ax.set_zlim(0, 1.0)
        ax.set_zlabel("|sensitivity|")
        ax.view_init(elev=30, azim=-55)
        row_vals = ", ".join(
            f"{instruments[c]}={M[active_row,c]:+.2f}"
            for c in range(nI) if abs(M[active_row, c]) > 0.01
        )
        _hud(ax, [
            f"Selected event = {events[active_row]}",
            f"  {row_vals}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "sensitivity_matrix")


# ---------------------------------------------------------------------
# 16. Surprise Z-Score Engine
# ---------------------------------------------------------------------
def surprise_z_score():
    """Bell curve in 3D with 7 colored buckets. A marker (the actual
    release) sweeps across z-scores -3 → +3."""
    print("surprise_z_score...")
    z = np.linspace(-3.5, 3.5, 200)
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

    # Bucket boundaries: -2, -1, -0.5, +0.5, +1, +2
    bucket_edges = [-3.5, -2, -1, -0.5, 0.5, 1, 2, 3.5]
    bucket_colors = ["#8b0000", "#ff4444", "#ffbb33", "#9E9E9E",
                     "#cddc39", "#4CAF50", "#2e7d32"]
    bucket_labels = ["large_miss", "miss", "slight_miss", "inline",
                     "slight_beat", "beat", "large_beat"]

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Surprise Z-Score Buckets (large_miss ↔ large_beat)")
        # Animated: marker walks across z-axis -2.5 → +2.5
        z_now = -2.5 + (frame / FRAMES) * 5
        # Find bucket index
        bidx = 0
        for i in range(len(bucket_edges) - 1):
            if bucket_edges[i] <= z_now < bucket_edges[i + 1]:
                bidx = i
                break
        # Plot per-bucket colored ribbons (z, height=pdf, depth)
        for i in range(len(bucket_edges) - 1):
            mask = (z >= bucket_edges[i]) & (z < bucket_edges[i + 1])
            if not mask.any():
                continue
            z_seg = z[mask]
            p_seg = pdf[mask]
            depth = np.full_like(z_seg, 0)
            color = bucket_colors[i]
            alpha = 0.95 if i == bidx else 0.4
            ax.plot(z_seg, depth, p_seg, color=color, lw=3, alpha=alpha)
            # Filled "wall" via bar3d-style ribbon
            for zs, ps in zip(z_seg[::3], p_seg[::3]):
                ax.bar3d([zs], [-0.2], [0], 0.05, 0.4, ps,
                         color=color, alpha=alpha * 0.5, shade=False)
        # Marker (the current release)
        z_pdf = np.exp(-0.5 * z_now**2) / np.sqrt(2 * np.pi)
        ax.scatter([z_now], [0], [z_pdf], color="white",
                   s=120, edgecolor="#00E5FF", linewidth=2)
        ax.plot([z_now, z_now], [0, 0], [0, z_pdf],
                color="#00E5FF", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Z-Score")
        ax.set_yticks([])
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.5)
        ax.set_zlabel("Density")
        ax.view_init(elev=25, azim=-55)
        _hud(ax, [
            f"Actual − Consensus = {z_now:+.2f}σ",
            f"Bucket             = {bucket_labels[bidx]}",
            f"P(≥ this z)        = {(1 - 0.5*(1+np.tanh(z_now*0.798)))*100:4.1f}%",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "surprise_z_score")


# ---------------------------------------------------------------------
# 17. Event-Type Multipliers
# ---------------------------------------------------------------------
def event_type_multipliers():
    """Bars per event category showing the magnitude multiplier
    (1.5x inflation, 2.0x rates, 1.3x employment, 1.1x growth, 0.9x PMI).
    Animate one category highlighted at a time."""
    print("event_type_multipliers...")
    cats = ["Inflation", "Rates", "Employment", "Growth", "PMI"]
    mults = np.array([1.5, 2.0, 1.3, 1.1, 0.9])
    examples = ["CPI/PCE", "FOMC", "NFP/Unemp", "GDP/Retail", "ISM"]
    n = len(cats)
    xs = np.arange(n)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Event-Type Volatility Multipliers")
        active = int((frame / FRAMES) * n) % n
        colors = ["#9E9E9E"] * n
        colors[active] = "#FFC107"
        ax.bar3d(xs, np.zeros(n), np.zeros(n),
                 0.6, 0.6, mults,
                 color=colors, alpha=0.9, shade=True)
        for i, m in enumerate(mults):
            ax.text(i + 0.3, 0.3, m + 0.07, f"{m}×",
                    color="white", fontsize=9, ha="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_yticks([])
        ax.set_zlim(0, 2.4)
        ax.set_zlabel("Multiplier")
        ax.view_init(elev=25, azim=-55)
        _hud(ax, [
            f"Active category = {cats[active]}",
            f"Example events  = {examples[active]}",
            f"Multiplier      = {mults[active]:.1f}× baseline",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "event_type_multipliers")


# ---------------------------------------------------------------------
# 18. Cross-Asset Type Scaling
# ---------------------------------------------------------------------
def cross_asset_scaling():
    """A single 1σ surprise propagates through asset-class scaling
    factors. Bars: 4 asset classes. Animate: surprise size sweeps -2σ → +2σ."""
    print("cross_asset_scaling...")
    asset_classes = ["Equity", "Bonds", "FX", "Commodity"]
    type_mult = np.array([1.0, 0.7, 0.5, 0.8])
    colors_ac = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    n = len(asset_classes)
    xs = np.arange(n)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Cross-Asset Type Scaling for a 1σ Surprise")
        # Animated: surprise sweeps -2σ → +2σ
        z = -2 + (frame / FRAMES) * 4
        baseline = 1.1  # daily SPX σ
        moves = baseline * type_mult * z
        ax.bar3d(xs, np.zeros(n), np.zeros(n),
                 0.6, 0.6, moves,
                 color=colors_ac, alpha=0.9, shade=True)
        for i, m in enumerate(moves):
            ax.text(i + 0.3, 0.3, m + np.sign(m) * 0.15,
                    f"{m:+.2f}%", color="white",
                    fontsize=8, ha="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(asset_classes)
        ax.set_yticks([])
        ax.set_zlim(-3, 3)
        ax.set_zlabel("Expected Move %")
        ax.view_init(elev=25, azim=-55)
        _hud(ax, [
            f"Surprise z       = {z:+.2f}σ",
            f"Baseline (SPX σ) = {baseline:.2f}%",
            f"Scaling: eq 1.0 / fi 0.7 / fx 0.5 / cm 0.8",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "cross_asset_scaling")


# ---------------------------------------------------------------------
# 19. Market Regime Classifier
# ---------------------------------------------------------------------
def market_regime():
    """3D landscape with 4 quadrants: VIX vs yield-curve slope. Marker
    walks across the quadrants showing regime transitions."""
    print("market_regime...")
    vix = np.linspace(8, 40, 60)
    slope = np.linspace(-1.5, 2.5, 60)
    V, S = np.meshgrid(vix, slope)
    # Regime height: highest in the corners (extreme regimes)
    Z = np.tanh((V - 22) / 8)**2 + np.tanh(S / 1.2)**2

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Market Regime: VIX × Yield-Curve Slope")
        ax.plot_surface(V, S, Z, cmap=cm.RdYlGn_r, alpha=0.6,
                        linewidth=0, antialiased=True)
        # Animated marker tracing a path: low-vol expansion →
        # rising VIX → recession risk → recovery
        t = frame / FRAMES * 2 * np.pi
        vix_now = 22 + 12 * np.sin(t)
        slope_now = 0.8 - 1.6 * np.sin(t * 0.5)
        z_now = (np.tanh((vix_now - 22) / 8)**2
                 + np.tanh(slope_now / 1.2)**2)
        ax.scatter([vix_now], [slope_now], [z_now + 0.05],
                   color="#00E5FF", s=120, edgecolor="white", linewidth=1.5)
        ax.plot([vix_now, vix_now], [slope_now, slope_now], [0, z_now + 0.05],
                color="#00E5FF", linewidth=1.5, linestyle="--")
        ax.set_xlabel("VIX")
        ax.set_ylabel("Curve slope (10Y-3M, %)")
        ax.set_zlabel("Regime stress")
        ax.set_zlim(0, 2.2)
        ax.view_init(elev=28, azim=-60)
        # Classify
        risk = "RISK_OFF" if vix_now > 25 else "RISK_ON" if vix_now < 15 else "NEUTRAL"
        growth = "RECESSION_RISK" if slope_now < 0 else "EXPANSION"
        _hud(ax, [
            f"VIX        = {vix_now:5.2f}",
            f"Slope 10-3 = {slope_now:+.2f}%",
            f"Risk regime    = {risk}",
            f"Growth regime  = {growth}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "market_regime")


# ---------------------------------------------------------------------
# 20. Implied-Expectation Reverse-Engineering
# ---------------------------------------------------------------------
def implied_expectation():
    """Pre-release moves across instruments → inferred consensus z-score
    via inverse sensitivity. Animate: market moves change → inferred
    z updates."""
    print("implied_expectation...")
    instruments = ["SPY", "QQQ", "TLT", "DXY", "EUR", "GLD"]
    # Sensitivities for CPI MoM (from SENSITIVITY_MATRIX)
    sens = np.array([-0.40, -0.55, -0.60, 0.25, -0.22, -0.35])
    n = len(instruments)
    xs = np.arange(n)

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Implied Consensus: Reverse-Engineered from Pre-Release Moves")
        # Animated true z-score that the market is slowly pricing in
        true_z = 1.5 * np.sin(frame / FRAMES * 2 * np.pi)
        # Pre-release moves = sensitivity × true_z + small noise
        rng = np.random.default_rng(frame)
        moves = sens * true_z + rng.normal(0, 0.05, n)
        # Inferred z per instrument (inverse) and aggregate
        inferred_per = moves / sens
        inferred_z = np.mean(inferred_per)
        # Bars: pre-release moves
        bar_colors = ["#4CAF50" if m > 0 else "#ff4444" for m in moves]
        ax.bar3d(xs, np.zeros(n), np.zeros(n),
                 0.6, 0.6, moves,
                 color=bar_colors, alpha=0.9, shade=True)
        for i, m in enumerate(moves):
            ax.text(i + 0.3, 0.3, m + np.sign(m) * 0.05,
                    f"{m:+.2f}%", color="white",
                    fontsize=7, ha="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(instruments)
        ax.set_yticks([])
        ax.set_zlim(-1.2, 1.2)
        ax.set_zlabel("Pre-release move %")
        ax.view_init(elev=25, azim=-55)
        _hud(ax, [
            f"True surprise z    = {true_z:+.2f}σ  (hidden)",
            f"Inferred z (avg)   = {inferred_z:+.2f}σ",
            f"Confidence         = HIGH (n={n} instruments)",
            f"Implied direction  = {'BEAT' if inferred_z>0 else 'MISS'}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "implied_expectation")


# ---------------------------------------------------------------------
# 21. Combined Multi-Surprise Impact
# ---------------------------------------------------------------------
def combined_surprises():
    """Stacked 3D bars per instrument: contributions from CPI MoM, Core
    CPI MoM, and Retail Sales (all released same morning). Animate:
    events stack on one at a time."""
    print("combined_surprises...")
    instruments = ["SPY", "QQQ", "TLT", "DXY"]
    n = len(instruments)
    xs = np.arange(n)
    # Per-event impacts (move % at the same z=+1 surprise)
    cpi_imp     = np.array([-0.40, -0.55, -0.60,  0.25])
    core_imp    = np.array([-0.45, -0.60, -0.65,  0.28])
    retail_imp  = np.array([ 0.25,  0.30,  0.05,  0.15])

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Combined Impact: 3 Releases Same Morning")
        # Stage: at frame=0 only CPI; mid: +Core CPI; late: +Retail
        stage = (frame / FRAMES) * 3
        s_cpi    = min(stage, 1.0)
        s_core   = max(0, min(stage - 1, 1.0))
        s_retail = max(0, min(stage - 2, 1.0))
        # Stack each layer
        ax.bar3d(xs, np.zeros(n), np.zeros(n),
                 0.5, 0.5, cpi_imp * s_cpi,
                 color="#ff4444", alpha=0.9, shade=True)
        ax.bar3d(xs, np.zeros(n), cpi_imp * s_cpi,
                 0.5, 0.5, core_imp * s_core,
                 color="#FFC107", alpha=0.9, shade=True)
        ax.bar3d(xs, np.zeros(n),
                 cpi_imp * s_cpi + core_imp * s_core,
                 0.5, 0.5, retail_imp * s_retail,
                 color="#4CAF50", alpha=0.9, shade=True)
        # Total label
        total = cpi_imp * s_cpi + core_imp * s_core + retail_imp * s_retail
        for i, t in enumerate(total):
            ax.text(i + 0.25, 0.25, t + np.sign(t) * 0.08,
                    f"{t:+.2f}%", color="white",
                    fontsize=8, ha="center", weight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(instruments)
        ax.set_yticks([])
        ax.set_zlim(-2.0, 1.5)
        ax.set_zlabel("Combined move %")
        ax.view_init(elev=25, azim=-55)
        active = []
        if s_cpi > 0:    active.append("CPI MoM")
        if s_core > 0:   active.append("Core CPI")
        if s_retail > 0: active.append("Retail Sales")
        _hud(ax, [
            f"Active releases = {', '.join(active)}",
            f"SPY total       = {total[0]:+.2f}%",
            f"TLT total       = {total[2]:+.2f}%",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "combined_surprises")


# ---------------------------------------------------------------------
# 22. Cross-Instrument Correlation Matrix
# ---------------------------------------------------------------------
def correlation_matrix():
    """Rolling correlation 3D heatmap. Animate: rolling window slides
    through time, correlations shift as regime changes."""
    print("correlation_matrix...")
    instruments = ["SPY", "QQQ", "TLT", "DXY", "EUR", "GLD"]
    n = len(instruments)
    X, Y = np.meshgrid(np.arange(n), np.arange(n))

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        _setup_3d(ax, "Cross-Instrument Rolling Correlation Matrix")
        phase = frame / FRAMES * 2 * np.pi
        # Synthetic regime shift: in risk-on, SPY-QQQ ~+0.9, SPY-TLT ~-0.2,
        # in risk-off, correlations compress toward +1 (everything sells)
        risk_off = 0.5 + 0.4 * np.sin(phase)
        # Build correlation matrix
        C = np.array([
            [ 1.0,  0.92,  -0.20 + risk_off*0.4,  -0.30,  -0.55, -0.10],
            [ 0.92, 1.0,   -0.25 + risk_off*0.4,  -0.28,  -0.50, -0.05],
            [-0.20+risk_off*0.4, -0.25+risk_off*0.4, 1.0, -0.10, 0.10, 0.30],
            [-0.30, -0.28, -0.10, 1.0,  -0.85,  0.05],
            [-0.55, -0.50,  0.10, -0.85, 1.0,   0.05],
            [-0.10, -0.05,  0.30,  0.05, 0.05,  1.0],
        ])
        # Render as bars
        for i in range(n):
            for j in range(n):
                v = C[i, j]
                color = cm.RdYlGn((v + 1) / 2)
                ax.bar3d(j, i, 0, 0.7, 0.7, abs(v) + 0.01,
                         color=color, alpha=0.9, shade=True)
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(instruments, fontsize=7)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(instruments, fontsize=7)
        ax.set_zlim(0, 1.1)
        ax.set_zlabel("|correlation|")
        ax.view_init(elev=35, azim=-60)
        regime = "RISK_OFF" if risk_off > 0.7 else "RISK_ON" if risk_off < 0.3 else "NEUTRAL"
        _hud(ax, [
            f"Regime          = {regime}",
            f"SPY-TLT corr    = {C[0,2]:+.2f}",
            f"SPY-QQQ corr    = {C[0,1]:+.2f}",
            f"DXY-EUR corr    = {C[3,4]:+.2f}",
        ])
        return ()

    anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    _save(fig, anim, "correlation_matrix")


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
    fed_funds_futures()
    tips_spreads()
    vix_term_structure()
    fed_funds_target_rate()
    consensus_override()
    sensitivity_matrix()
    surprise_z_score()
    event_type_multipliers()
    cross_asset_scaling()
    market_regime()
    implied_expectation()
    combined_surprises()
    correlation_matrix()
    print("\nDone.")


if __name__ == "__main__":
    main()
