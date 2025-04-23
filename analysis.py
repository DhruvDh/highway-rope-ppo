# analysis.py
import re
import glob
import pandas as pd
import numpy as np


def load_and_parse(pattern="artifacts/combined_validated_data*.csv"):
    """Loads and parses the combined CSV data, extracting hyperparameters."""
    csv_files = glob.glob(pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found matching pattern: {pattern}")
    path = csv_files[0]
    if len(csv_files) > 1:
        print(f"Warning: Multiple files matched pattern '{pattern}'. Using '{path}'.")

    records = []
    # --- CORRECTED REGEX ---
    # Capture everything after 'feat=' greedily until the mandatory '_epochs=' part.
    # This handles commas and underscores within the feature names correctly.
    rx = re.compile(
        r"^feat=(?P<features>.+)"
        r"_epochs=(?P<epochs>\d+)"
        r"_lr=(?P<lr>[0-9.eE+-]+)"
        r"_hidden_dim=(?P<hidden_dim>\d+)"
        r"_batch_size=(?P<batch_size>\d+)$"
    )
    # --- END OF CORRECTION ---

    print(f"Loading data from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline()  # Read header line

            for line_num, line in enumerate(f, 2):
                line = line.strip()
                if not line:
                    continue
                tokens = line.split(",")

                landmark_token_part = "_epochs="
                landmark_idx = next(
                    (i for i, t in enumerate(tokens) if landmark_token_part in t), None
                )

                if landmark_idx is None or len(tokens) < landmark_idx + 4:
                    print(
                        f"Warning: Skipping line {line_num}. Could not find landmark '{landmark_token_part}' or insufficient fields after it. Line: '{line}'"
                    )
                    continue

                exp_name = ",".join(tokens[: landmark_idx + 1])

                try:
                    final = float(tokens[landmark_idx + 1])
                    _max = float(tokens[landmark_idx + 2])
                    steps = int(tokens[landmark_idx + 3])
                except (ValueError, IndexError):
                    print(
                        f"Warning: Skipping line {line_num}. Error parsing numeric fields expected after landmark. Line: '{line}'"
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
        df = df.astype(
            {
                "epochs": "int",
                "lr": "float",
                "hidden_dim": "int",
                "batch_size": "int",
            }
        )
    except KeyError as e:
        print(
            f"Error converting column types. Missing column: {e}. DataFrame columns: {df.columns}"
        )
        return pd.DataFrame()

    print(
        f"Successfully loaded and parsed {len(df)} records."
    )  # Now reflects records *not* skipped by regex
    return df


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

    print("\n=== By features (Full Dataset) ===")
    if "features" in df.columns:
        print(
            df.groupby("features")["final_reward"]
            .agg(["mean", "std", "count"])
            .round(2)
        )
    else:
        print("Column 'features' not found for grouping.")

    print("\n=== By learning rate (Full Dataset) ===")
    if "lr" in df.columns:
        print(df.groupby("lr")["final_reward"].agg(["mean", "std", "count"]).round(2))
    else:
        print("Column 'lr' not found for grouping.")

    print("\n=== By epochs/update (Full Dataset) ===")
    if "epochs" in df.columns:
        print(
            df.groupby("epochs")["final_reward"].agg(["mean", "std", "count"]).round(2)
        )
    else:
        print("Column 'epochs' not found for grouping.")

    print("\n=== By hidden_dim (Full Dataset) ===")
    if "hidden_dim" in df.columns:
        print(
            df.groupby("hidden_dim")["final_reward"]
            .agg(["mean", "std", "count"])
            .round(2)
        )
    else:
        print("Column 'hidden_dim' not found for grouping.")

    print("\n=== By batch_size (Full Dataset) ===")
    if "batch_size" in df.columns:
        print(
            df.groupby("batch_size")["final_reward"]
            .agg(["mean", "std", "count"])
            .round(2)
        )
    else:
        print("Column 'batch_size' not found for grouping.")

    # --- New Analysis for Fixed Final Config Subset ---
    print("\n\n" + "=" * 15 + " Analysis for Fixed Final Config Subset " + "=" * 15)

    fixed_features = "x,y,vx,vy"
    fixed_lr = 3e-4
    fixed_epochs = 8

    print("\nFiltering criteria for this subset:")
    print(f"  - features = '{fixed_features}'")
    print(f"  - lr       = {fixed_lr}")
    print(f"  - epochs   = {fixed_epochs}")
    print(
        "(Note: clip_eps and entropy_coef are not available in the loaded data for filtering)"
    )

    required_cols = ["features", "lr", "epochs"]
    if not all(col in df.columns for col in required_cols):
        print(
            "\n*** Cannot perform filtering. Required columns missing from DataFrame. ***"
        )
        print(f"    Required: {required_cols}")
        print(f"    Available: {list(df.columns)}")
        return

    df_filtered = df[
        (df["features"] == fixed_features)
        & (np.isclose(df["lr"], fixed_lr))
        & (df["epochs"] == fixed_epochs)
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
