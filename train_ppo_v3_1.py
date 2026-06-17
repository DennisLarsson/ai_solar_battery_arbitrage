import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from v5_4.solar_battery_env_v5_4 import SolarBatteryEnv
import prepare_smhi_datasets as sp

def run_ppo_training(pv_file = "../../regression/smhi_2025_predicted_pv_output_with_time.csv",
                     sp_file = "../../data/energy-charts_Electricity_production_and_spot_prices_in_Sweden_in_2025.csv",
                     ppo_model_save_path = "/home/dennis/Documents/machine-learning/Solar/RL_agent/solar_battery_ppo_v3.zip",
                     verbose=1,
                     seed=None,
                     battery_capacity=8.0,
                     max_pv=1.0,
                     forecast_horizon = 36,
                     learning_rate=3e-4,
                     n_steps=2048,
                     batch_size=64,
                     gamma=0.99,
                     gae_lambda=0.95,
                     ent_coef=0.0,
                     n_epochs=10,
                     timesteps_multiplier=100_000
                     ):

    smhi_pv_pred_td = sp.read_smhi_predicted_pv_output(pv_file)
    spotprices = sp.read_energy_chart_spotprices(sp_file)

    #if not sp.are_datetimes_identical(smhi_pv_pred_td, spotprices):
    #    sys.exit("The times and dates of PV data and price data is not the same!")

    pv_forecast = smhi_pv_pred_td['P'].values / 1000
    price_forecast = spotprices['price'].values / 1000


    if len(pv_forecast) != len(price_forecast):
        print("length pv_forecast:", len(pv_forecast), "length price_forecast:", len(price_forecast))
        sys.exit("Length of pv_forecast and price_forecast do not match")

    # Total length minus forecast horizon
    #total_hours = len(pv_forecast)
    #print("Total hours: ", total_hours)

    def make_env(start_idx):
        """Factory function to create an env with a specific start index"""
        def _init():
            pv_slice = pv_forecast[start_idx : start_idx + forecast_horizon]
            price_slice = price_forecast[start_idx : start_idx + forecast_horizon]
            env = SolarBatteryEnv(pv_slice, price_slice, battery_capacity=battery_capacity, max_pv=max_pv)
            return env
        return _init

    start_indices = np.array([], dtype=int)
    index = int(14)
    while index < len(price_forecast)-forecast_horizon:
        start_indices = np.append(start_indices, index)
        index = int(index + 24)

    envs = [make_env(i) for i in start_indices]

    #print("Nr of envs: ", len(envs))
    #print("length envs: ", [len(env()._get_obs()) for env in envs])

    total_timesteps = len(envs)*timesteps_multiplier
    #print("Total timesteps: ", total_timesteps)

    # Wrap in DummyVecEnv
    vec_env = DummyVecEnv(envs)

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        n_epochs=n_epochs,
        seed=seed,
        device='cpu'
    )

    # Train for MUCH longer (e.g., 500k or 1M steps)
    # Because we have 10 parallel environments, 100k steps = 1M environment steps
    model.learn(total_timesteps=total_timesteps)

    # Save
    model.save(ppo_model_save_path)
