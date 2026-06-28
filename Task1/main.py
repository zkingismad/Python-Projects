import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#2e3149",
    "axes.labelcolor": "#c8ccf0",
    "axes.titlecolor": "#e8eaf6",
    "xtick.color": "#8a8fbb",
    "ytick.color": "#8a8fbb",
    "text.color": "#c8ccf0",
    "grid.color": "#2e3149",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#2e3149",
    "font.family": "sans-serif",
})

ACCENT = ["#7c83f5", "#f5a623", "#50fa7b", "#ff6b6b", "#a78bfa", "#38bdf8"]
OUTPUT_DIR = "C:/Customization/Programming/Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("C:/Users/ishme/Downloads/teams.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape          : {df.shape[0]} teams x {df.shape[1]} columns")
print(f"Teams          : {list(df['team'].values)}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"\nNumeric columns: {len(numeric_cols)}")

print("\n" + "=" * 60)
print("KEY COLUMN AVERAGES")
print("=" * 60)
key_cols = [
    "goals", "goals_against", "possession", "shots",
    "shots_on_target_pct", "gk_save_pct", "gk_clean_sheets_pct",
    "plus_minus", "points_per_game", "goals_per90",
]
for col in key_cols:
    if col in df.columns:
        print(f"  Avg {col:<30}: {df[col].mean():>8.2f}")

print("\n" + "=" * 60)
print("TOP 5 TEAMS — GOALS SCORED")
print("=" * 60)
print(df[["team", "goals"]].sort_values("goals", ascending=False).head(5).to_string(index=False))

print("\n" + "=" * 60)
print("TOP 5 TEAMS — POINTS PER GAME")
print("=" * 60)
print(df[["team", "points_per_game", "goals", "goals_against"]]
      .sort_values("points_per_game", ascending=False).head(5).to_string(index=False))

print("\n" + "=" * 60)
print("BEST DEFENCES (fewest goals against)")
print("=" * 60)
print(df[["team", "goals_against", "gk_save_pct", "gk_clean_sheets_pct"]]
      .sort_values("goals_against").head(5).to_string(index=False))

print("\n" + "=" * 60)
print("POSSESSION LEADERS")
print("=" * 60)
print(df[["team", "possession", "goals", "shots"]]
      .sort_values("possession", ascending=False).head(5).to_string(index=False))

df = df.copy()
df["goal_diff"] = df["goals"] - df["goals_against"]
df["shot_accuracy"] = df["shots_on_target_pct"].fillna(0)


fig, axes = plt.subplots(2, 2, figsize=(17, 13))
fig.suptitle("International Football Team Analysis", fontsize=18,
             fontweight="bold", color="#e8eaf6", y=0.99)
fig.patch.set_facecolor("#0f1117")

ax1 = axes[0, 0]
top10 = df.nlargest(10, "goals")[["team", "goals", "goals_against"]].sort_values("goals")
x = np.arange(len(top10))
w = 0.38
ax1.barh(x - w/2, top10["goals"], w, color="#7c83f5", label="Goals Scored", zorder=3)
ax1.barh(x + w/2, top10["goals_against"], w, color="#ff6b6b", label="Goals Conceded", zorder=3)
ax1.set_yticks(x)
ax1.set_yticklabels(top10["team"], fontsize=8)
ax1.set_title("Top 10 Teams: Goals Scored vs Conceded", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Goals")
ax1.legend(fontsize=9)
ax1.grid(axis="x", zorder=0)

ax2 = axes[0, 1]
sc = ax2.scatter(
    df["possession"], df["goals"],
    c=df["points_per_game"], cmap="plasma",
    s=100, alpha=0.85, edgecolors="#0f1117", linewidths=0.6, zorder=3
)
cbar = fig.colorbar(sc, ax=ax2)
cbar.set_label("Points per Game", color="#c8ccf0")
cbar.ax.yaxis.set_tick_params(color="#c8ccf0")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c8ccf0")
for _, row in df.nlargest(5, "goals").iterrows():
    ax2.annotate(row["team"], xy=(row["possession"], row["goals"]),
                 xytext=(5, 4), textcoords="offset points", fontsize=7.5, color="#c8ccf0")
m, b = np.polyfit(df["possession"].dropna(), df["goals"].dropna(), 1)
xr = np.linspace(df["possession"].min(), df["possession"].max(), 100)
ax2.plot(xr, m*xr + b, color="#50fa7b", linewidth=1.8, linestyle="--", label="Trend", zorder=4)
ax2.set_title("Possession vs Goals (colour = PPG)", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Possession %")
ax2.set_ylabel("Goals Scored")
ax2.legend(fontsize=9)
ax2.grid(zorder=0)

ax3 = axes[1, 0]
heat_cols = ["goals", "goals_against", "possession",
             "shots_on_target_pct", "gk_save_pct", "plus_minus", "points_per_game"]
heat_cols = [c for c in heat_cols if c in df.columns]
corr = df[heat_cols].corr().round(2)
im = ax3.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
fig.colorbar(im, ax=ax3, shrink=0.8, label="Correlation")
ax3.set_xticks(range(len(heat_cols)))
ax3.set_yticks(range(len(heat_cols)))
labels = [c.replace("_", "\n") for c in heat_cols]
ax3.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
ax3.set_yticklabels(labels, fontsize=7)
for i in range(len(heat_cols)):
    for j in range(len(heat_cols)):
        ax3.text(j, i, f"{corr.values[i, j]:.2f}",
                 ha="center", va="center", fontsize=7.5,
                 color="white", fontweight="bold")
ax3.set_title("Correlation Heatmap — Key Metrics", fontsize=12, fontweight="bold", pad=10)

ax4 = axes[1, 1]
top15 = df.nlargest(15, "points_per_game").sort_values("points_per_game")
colors = ["#50fa7b" if v >= 2.0 else "#7c83f5" if v >= 1.5 else "#f5a623"
          for v in top15["points_per_game"]]
bars = ax4.barh(top15["team"], top15["points_per_game"],
                color=colors, zorder=3, height=0.65)
ax4.set_title("Top 15 Teams by Points per Game", fontsize=12, fontweight="bold", pad=10)
ax4.set_xlabel("Points per Game")
ax4.axvline(x=top15["points_per_game"].mean(), color="#ff6b6b",
            linestyle="--", linewidth=1.5, label="Avg PPG", zorder=4)
ax4.legend(fontsize=9)
ax4.grid(axis="x", zorder=0)
for bar, val in zip(bars, top15["points_per_game"]):
    ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=8, color="#e8eaf6")
ax4.tick_params(axis="y", labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
dash_path = os.path.join(OUTPUT_DIR, "teams_dashboard.png")
plt.savefig(dash_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print(f"\n[Saved] Dashboard -> {dash_path}")


fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Attacking vs Defensive Efficiency", fontsize=15,
              fontweight="bold", color="#e8eaf6")
fig2.patch.set_facecolor("#0f1117")

ax5 = axes2[0]
sc2 = ax5.scatter(df["shots"], df["goals"],
                  c=df["shot_accuracy"], cmap="YlOrRd",
                  s=90, alpha=0.85, edgecolors="#0f1117", linewidths=0.5, zorder=3)
cbar2 = fig2.colorbar(sc2, ax=ax5)
cbar2.set_label("Shot Accuracy %", color="#c8ccf0")
cbar2.ax.yaxis.set_tick_params(color="#c8ccf0")
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#c8ccf0")
for _, row in df.nlargest(6, "goals").iterrows():
    ax5.annotate(row["team"], xy=(row["shots"], row["goals"]),
                 xytext=(4, 4), textcoords="offset points", fontsize=7.5, color="#c8ccf0")
ax5.set_title("Shots vs Goals (colour = accuracy %)", fontsize=11, fontweight="bold", pad=10)
ax5.set_xlabel("Total Shots")
ax5.set_ylabel("Goals Scored")
ax5.grid(zorder=0)

ax6 = axes2[1]
clean_df = df[["team", "gk_save_pct", "gk_clean_sheets_pct", "goals_against"]]\
    .dropna().sort_values("gk_save_pct", ascending=False).head(12)
x2 = np.arange(len(clean_df))
w2 = 0.38
ax6.bar(x2 - w2/2, clean_df["gk_save_pct"], w2, color="#38bdf8", label="Save %", zorder=3)
ax6.bar(x2 + w2/2, clean_df["gk_clean_sheets_pct"] * 100, w2,
        color="#a78bfa", label="Clean Sheet %", zorder=3)
ax6.set_xticks(x2)
ax6.set_xticklabels(clean_df["team"], rotation=35, ha="right", fontsize=7.5)
ax6.set_title("Goalkeeper: Save % vs Clean Sheet %", fontsize=11, fontweight="bold", pad=10)
ax6.set_ylabel("Percentage")
ax6.legend(fontsize=9)
ax6.grid(axis="y", zorder=0)

plt.tight_layout()
eff_path = os.path.join(OUTPUT_DIR, "teams_efficiency.png")
plt.savefig(eff_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print(f"[Saved] Efficiency Chart -> {eff_path}")


print("\n" + "=" * 60)
print("KEY INSIGHTS & OBSERVATIONS")
print("=" * 60)
best_team = df.loc[df["points_per_game"].idxmax(), "team"]
print(f"\n1. Best performing team : {best_team} ({df['points_per_game'].max():.2f} pts/game)")
top3 = df.nlargest(3, "goals")[["team", "goals"]]
print(f"2. Top scorers          : " + ", ".join(f"{r.team} ({r.goals})" for _, r in top3.iterrows()))
best_def = df.loc[df["goals_against"].idxmin(), "team"]
print(f"3. Stingiest defence    : {best_def} ({df['goals_against'].min()} conceded)")
poss_corr = df[["possession", "goals"]].corr().iloc[0, 1]
print(f"4. Possession-Goals correlation: {poss_corr:.2f}")
best_gk = df.loc[df["gk_save_pct"].idxmax(), ["team", "gk_save_pct"]]
print(f"5. Best save percentage : {best_gk['team']} ({best_gk['gk_save_pct']:.1f}%)")
print(f"6. Avg goals scored / conceded : {df['goals'].mean():.1f} / {df['goals_against'].mean():.1f}")
print(f"7. Avg possession across teams : {df['possession'].mean():.1f}%")
print(f"\nCharts saved to the '{OUTPUT_DIR}' folder.")