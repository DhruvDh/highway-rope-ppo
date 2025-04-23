# analysis.py
import re
import glob
import pandas as pd
import numpy as np


def load_and_parse(pattern="artifacts/combined_validated_data-final-run.csv"):
    """Loads and parses the combined CSV data, extracting hyperparameters."""
    csv_files = glob.glob(pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found matching pattern: {pattern}")
    path = csv_files[0]
    if len(csv_files) > 1:
        print(f"Warning: Multiple files matched pattern '{pattern}'. Using '{path}'.")

    records = []
    # --- UPDATED REGEX FOR FINAL RUN ---
    # Matches names like:
    #   shuffled_rope_lr0.0003_hidden_dim256_clip_eps0.2_entropy_coef0.005_epochs8_batch_size64_d_embed16_seed2042
    rx = re.compile(
        r"^(?P<prefix>sorted|shuffled)"
        r"(?:_(?P<pe_type>rankpe|distpe|rope))?"  # optional PE subtype
        r"_lr(?P<lr>[0-9.eE+-]+)"
        r"_hidden_dim(?P<hidden_dim>\d+)"
        r"_clip_eps(?P<clip_eps>[0-9.eE+-]+)"
        r"_entropy_coef(?P<entropy_coef>[0-9.eE+-]+)"
        r"_epochs(?P<epochs>\d+)"
        r"_batch_size(?P<batch_size>\d+)"
        r"_d_embed(?P<d_embed>\d+)"
        r"_seed(?P<seed>\d+)$"
    )
    # --- END UPDATED REGEX ---

    print(f"Loading data from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.readline()  # Read header line

            for line_num, line in enumerate(f, 2):
                line = line.strip()
                if not line:
                    continue
                tokens = line.split(",")

                # Expected CSV layout:
                #   0: experiment_name
                #   1: final_reward
                #   2: max_reward
                #   3: training_steps
                #   4: best_model_path
                #   5: plot_path
                if len(tokens) < 4:
                    print(
                        f"Warning: Skipping line {line_num}. Expected ≥4 comma‑separated fields, got {len(tokens)}. Line: '{line}'"
                    )
                    continue

                exp_name = tokens[0]
                try:
                    final = float(tokens[1])
                    _max = float(tokens[2])
                    steps = int(tokens[3])
                except ValueError:
                    print(
                        f"Warning: Skipping line {line_num}. Failed to parse numeric fields. Line: '{line}'"
                    )
                    continue

                match = rx.match(exp_name)  # Apply the corrected regex
                if match:
                    hyp = match.groupdict()
                    records.append(
                        {
                            "experiment": exp_name,
                            "final_reward": final,
                            "max_reward": _max,
                            "training_steps": steps,
                            **hyp,
                        }
                    )
                else:
                    # This warning should no longer appear for valid lines with the corrected regex
                    print(
                        f"Warning: Skipping line {line_num} as reconstructed experiment name '{exp_name}' did not match regex."
                    )

    except FileNotFoundError:
        print(f"Error: Could not find the data file at {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading {path}: {e}")
        return pd.DataFrame()

    if not records:
        print("No valid records were parsed from the CSV file.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    try:
        astype_cols = {
            "epochs": "int",
            "lr": "float",
            "hidden_dim": "int",
            "batch_size": "int",
            "clip_eps": "float",
            "entropy_coef": "float",
            "d_embed": "int",
            "seed": "int",
        }
        df = df.astype({k: v for k, v in astype_cols.items() if k in df.columns})
    except KeyError as e:
        print(
            f"Error converting column types. Missing column: {e}. DataFrame columns: {df.columns}"
        )
        return pd.DataFrame()

    print(
        f"Successfully loaded and parsed {len(df)} records."
    )  # Now reflects records *not* skipped by regex
    return df


def _print_group(df: pd.DataFrame, col: str, title: str) -> None:
    """Utility to print grouped mean/std/count if column exists."""
    print(f"\n=== By {title} (Full Dataset) ===")
    if col in df.columns:
        print(df.groupby(col)["final_reward"].agg(["mean", "std", "count"]).round(2))
    else:
        print(f"Column '{col}' not found for grouping.")


# --- Main function remains unchanged ---
def main():
    df = load_and_parse()

    if df.empty:
        print("Exiting due to data loading issues.")
        return

    # --- Original Analysis (Full Dataset) ---
    print("\n\n" + "=" * 15 + " Analysis on Full Midterm Dataset " + "=" * 15)

    print("\n=== Overall metrics (Full Dataset) ===")
    print(df[["final_reward", "max_reward", "training_steps"]].describe().round(2))

    _print_group(df, "features", "features")
    _print_group(df, "lr", "learning rate")
    _print_group(df, "epochs", "epochs/update")
    _print_group(df, "hidden_dim", "hidden_dim")
    _print_group(df, "batch_size", "batch_size")

    # --- New Analysis for Fixed Final Config Subset ---
    print("\n\n" + "=" * 15 + " Analysis for Fixed Final Config Subset " + "=" * 15)

    fixed_lr = 3e-4
    fixed_epochs = 8

    print("\nFiltering criteria for this subset:")
    print(f"  - lr       = {fixed_lr}")
    print(f"  - epochs   = {fixed_epochs}")
    print(
        "(Note: clip_eps and entropy_coef are not available in the loaded data for filtering)"
    )

    df_filtered = df[
        (np.isclose(df["lr"], fixed_lr)) & (df["epochs"] == fixed_epochs)
    ].copy()

    if df_filtered.empty:
        print(
            "\n*** No data found matching the fixed configuration criteria in the loaded dataset. ***"
        )
        print("*** Cannot perform analysis on the fixed subset. ***")
    else:
        print(f"\nFound {len(df_filtered)} runs matching the fixed criteria.")

        print("\n--- Overall metrics (Filtered Subset) ---")
        print(
            df_filtered[["final_reward", "max_reward", "training_steps"]]
            .describe()
            .round(2)
        )

        print("\n--- By hidden_dim (within Fixed Config) ---")
        if "hidden_dim" in df_filtered.columns:
            print(
                df_filtered.groupby("hidden_dim")["final_reward"]
                .agg(["mean", "std", "count"])
                .round(2)
            )
        else:
            print("Column 'hidden_dim' not found for grouping in filtered data.")

        print("\n--- By batch_size (within Fixed Config) ---")
        if "batch_size" in df_filtered.columns:
            print(
                df_filtered.groupby("batch_size")["final_reward"]
                .agg(["mean", "std", "count"])
                .round(2)
            )
        else:
            print("Column 'batch_size' not found for grouping in filtered data.")

        if not df_filtered.empty and "final_reward" in df_filtered.columns:
            best_run_filtered = df_filtered.loc[df_filtered["final_reward"].idxmax()]
            print("\n--- Best Performing Run (within Fixed Config Subset) ---")
            print(f"  Experiment Name: {best_run_filtered.get('experiment', 'N/A')}")
            print(
                f"  Final Reward:    {best_run_filtered.get('final_reward', float('nan')):.2f}"
            )
            print(f"  Hidden Dim:      {best_run_filtered.get('hidden_dim', 'N/A')}")
            print(f"  Batch Size:      {best_run_filtered.get('batch_size', 'N/A')}")


if __name__ == "__main__":
    main()
