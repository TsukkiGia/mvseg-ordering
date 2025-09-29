import pandas
from pathlib import Path
import matplotlib.pyplot as plt



def plot_image_index_vs_initial_dice(experiment_number, script_dir):
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    experiment_dir = results_dir/ f"Experiment_{experiment_number}"
    df = pandas.read_csv(experiment_dir / "all_image_results.csv")
    plt.figure(figsize=(12,6))
    for permutation, subdf in df.groupby("permutation_seed"):
        plt.plot(subdf["image_index"], subdf["initial_dice"], marker="o", label=f"Perm {permutation}")
    
    plt.xlabel("Image Index")
    plt.ylabel("Initial Dice")
    plt.title("Initial Dice per Image Across Permutations")
    plt.legend()

    figure_dir = experiment_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.savefig(figure_dir / "ImageIndexVsDice.png")

def plot_image_index_vs_iterations_used(experiment_number, script_dir):
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    experiment_dir = results_dir/ f"Experiment_{experiment_number}"
    df = pandas.read_csv(experiment_dir / "all_image_results.csv")
    plt.figure(figsize=(12,6))
    for permutation, subdf in df.groupby("permutation_seed"):
        plt.plot(subdf["image_index"], subdf["iterations_used"], marker="o", label=f"Perm {permutation}")

    plt.axhline(y=df["prompt_limit"].iloc[0], color="red", linestyle="--", label="Max prompts (failure)")
    plt.yticks(range(0, df["prompt_limit"].iloc[0] + 1))
    plt.xlabel("Image Index")
    plt.ylabel("Prompt Iterations used")
    plt.title("Iterations used per Image Across Permutations")
    plt.legend()

    figure_dir = experiment_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.savefig(figure_dir / "ImageIndexVsPromptIters.png")

def plot_boxplot_per_index(experiment_number, script_dir):
    results_dir = script_dir / "results"
    experiment_dir = results_dir / f"Experiment_{experiment_number}"
    df = pandas.read_csv(experiment_dir / "all_image_results.csv")

    plt.figure(figsize=(12,6))
    df.boxplot(column="initial_dice", by="image_index")
    plt.title("Distribution of Initial Dice Across Permutations")
    plt.suptitle("")  # remove pandas auto-title
    plt.xlabel("Image Index")
    plt.ylabel("Initial Dice")

    figure_dir = experiment_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.savefig(figure_dir / "BoxplotInitialDice.png")
    plt.close()

def plot_best_worst_gap(experiment_number, script_dir):
    results_dir = script_dir / "results"
    experiment_dir = results_dir / f"Experiment_{experiment_number}"
    df = pandas.read_csv(experiment_dir / "all_image_results.csv")

    grouped = df.groupby("image_index")["initial_dice"]
    min_dice = grouped.min()
    max_dice = grouped.max()
    gap = max_dice - min_dice

    plt.figure(figsize=(12,6))
    plt.plot(gap.index, gap.values, marker="o", color="red", label="Max–Min Gap")
    plt.xlabel("Image Index")
    plt.ylabel("Dice Range Across Permutations")
    plt.title("Spread Between Best and Worst Permutations")
    plt.legend()

    figure_dir = experiment_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.savefig(figure_dir / "BestWorstGap.png")
    plt.close()

# python -m experiments.analysis.results_plot
if __name__ == "__main__":
    dir = Path("/data/ddmg/mvseg-ordering/experiments/scripts/randomized_experiments")
    plot_best_worst_gap(0, dir)
    # script_dir = Path(__file__).resolve().parents[1]
    # for experiment_number in range(1):
    #     plot_image_index_vs_initial_dice(experiment_number, script_dir)
    #     plot_image_index_vs_iterations_used(experiment_number, script_dir)