import pandas
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
results_dir = PROJECT_ROOT / f"experiments/results"


def plot_image_index_vs_initial_dice(experiment_number):
    experiment_dir = results_dir/ f"Experiment {experiment_number}"
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

def plot_image_index_vs_iterations_used(experiment_number):
    experiment_dir = results_dir/ f"Experiment {experiment_number}"
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


if __name__ == "__main__":
    for experiment_number in range(1):
        plot_image_index_vs_initial_dice(experiment_number)
        plot_image_index_vs_iterations_used(experiment_number)