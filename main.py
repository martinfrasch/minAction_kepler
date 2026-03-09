
import torch
from data_kepler import OrbitConfig, generate_dataset, train_val_test_split
from train_minaction import main as train_main
from evaluate_minaction import plot_discovery_results

def run_full_experiment():
    print("--- Starting MinAction.Net Kepler Discovery Experiment ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model, energy_kwh, test_data = train_main()
    print("\n--- Running Evaluation ---")
    sample_orbit = {
        'r_clean': test_data['r_clean'][0],
        'v_clean': test_data['v_clean'][0],
        't': test_data['t'][0]
    }
    plot_discovery_results(model, sample_orbit, energy_kwh)

if __name__ == "__main__":
    run_full_experiment()
