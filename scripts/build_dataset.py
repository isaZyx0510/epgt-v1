from thesis_transformer_v1.data.config import load_experiment_config
from thesis_transformer_v1.data.generator import generate_thesis_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate one in-memory thesis dataset summary.")
    parser.add_argument("--config", default="configs/data/base_common_delay.yaml")
    args = parser.parse_args()
    data = generate_thesis_dataset(load_experiment_config(args.config))
    for key in ["tx_grid", "rx_grid", "h_freq"]:
        print(f"{key}: shape={data[key].shape} dtype={data[key].dtype}")
    labels = data["channel_labels"]
    for key in ["total_delay_s", "cfo_hz", "rx_time_offsets_s", "path_delays_s"]:
        print(f"channel_labels.{key}: shape={labels[key].shape}")


if __name__ == "__main__":
    main()

