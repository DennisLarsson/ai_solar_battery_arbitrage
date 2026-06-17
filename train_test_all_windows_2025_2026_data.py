import sys
import os

sys.path.insert(0, os.getcwd())

from train_ppo import run_ppo_training
from test_ppo import run_test_ppo

base_dir = os.getcwd()

pv_file_train = os.path.join(base_dir, "../regression/smhi_2025_predicted_pv_output_with_time.csv")
sp_file_train = os.path.join(base_dir,
                             "../data/energy-charts_Electricity_production_and_spot_prices_in_Sweden_in_2025.csv")

pv_file_test = os.path.join(base_dir, "../regression/smhi_2026_predicted_pv_output_with_time.csv")
sp_file_test = os.path.join(base_dir,
                            "../data/energy-charts_Electricity_production_and_spot_prices_in_Sweden_in_2026.csv")

ppo_model = "/home/dennis/Documents/machine-learning/Solar/RL_agent/solar_battery_ppo_v3_1.zip"

seed = 124
battery_capacity = 8.0
max_pv = 1.0

run_ppo_training(
    pv_file=pv_file_train,
    sp_file=sp_file_train,
    verbose=1,
    seed=seed,
    battery_capacity=battery_capacity,
    max_pv=max_pv,
    forecast_horizon = 36,
    ppo_model_save_path = ppo_model
)

run_test_ppo(
    pv_file=pv_file_test,
    sp_file=sp_file_test,
    forecast_horizon=36,
    battery_capacity=battery_capacity,
    max_pv=max_pv,
    version_prefix = "v3_1",
    seed_val = seed,
    ppo_model = ppo_model
)
