import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random.mtrand import random_sample
from pandas.core import sample
from scipy import stats
from stable_baselines3 import PPO

import prepare_smhi_datasets as sp
from v5_4.solar_battery_env_v5_4 import SolarBatteryEnv


def run_test_ppo(pv_file = "../../regression/smhi_2026_predicted_pv_output_with_time.csv",
                 sp_file = "../../data/energy-charts_Electricity_production_and_spot_prices_in_Sweden_in_2026.csv",
                 ppo_model = "/home/dennis/Documents/machine-learning/Solar/RL_agent/solar_battery_ppo_v3.zip",
                 version_prefix = "",
                 forecast_horizon = 36,
                 battery_capacity=8.0,
                 max_pv=1.0,
                 plot_vis = True,
                 seed_val = None
                 ):

    smhi_pv_pred_td = sp.read_smhi_predicted_pv_output(pv_file)
    smhi_pv_pred_td = smhi_pv_pred_td.iloc[:-1]

    spotprices = sp.read_energy_chart_spotprices(sp_file)
    spotprices = spotprices[spotprices['time'].dt.month < 6]

    #if not sp.are_datetimes_identical(smhi_pv_pred_td[0], spotprices[0]):
    #    sys.exit("The times and dates of PV data and price data is not the same!")

    pv_forecast = smhi_pv_pred_td['P'].values / 1000
    price_forecast = spotprices['price'].values / 1000

    if len(pv_forecast) != len(price_forecast):
        print("length pv_forecast:", len(pv_forecast), "length price_forecast:", len(price_forecast))
        sys.exit("Length of pv_forecast and price_forecast do not match")

    model = PPO.load(ppo_model, device='cpu')

    start_indices = np.array([], dtype=int)
    index = int(14)
    while index < len(price_forecast)-forecast_horizon:
        start_indices = np.append(start_indices, index)
        index = int(index + 24)

    nr_starts = len(start_indices)

    def evaluate_on_fixed_windows(model_or_strategy, pv_data, price_data, start_indices, strategy_name=None):
        rewards = []
        action_history = []
        reward_history = []
        soc_history = []
        price_history = []

        for i, start_idx in enumerate(start_indices):
            pv_slice = pv_data[start_idx: start_idx + forecast_horizon]
            price_slice = price_data[start_idx: start_idx + forecast_horizon]

            env = SolarBatteryEnv(pv_slice, price_slice, battery_capacity=battery_capacity, max_pv=max_pv)

            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                current_price = env.price_forecast[env.current_step]
                if isinstance(model_or_strategy, str):
                    if model_or_strategy == 'random':
                        action = env.action_space.sample()
                    elif model_or_strategy == 'greedy':
                        avg_price = np.mean(env.price_forecast)
                        store_frac = 1.0 if current_price < avg_price else 0.0
                        discharge = 1.0 if current_price > avg_price else 0.0
                        action = np.array([store_frac, discharge])
                    elif model_or_strategy == 'do_nothing':
                        action = np.array([0.0, 0.0])
                else:
                    action, _ = model_or_strategy.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                soc_history.append(env.soc)
                action_history.append(action)
                reward_history.append(reward)
                price_history.append(current_price)

                if env.current_step > forecast_horizon:
                    break

            rewards.append(episode_reward)

        return np.array(rewards), soc_history, action_history, reward_history, price_history

    rl_rewards, rl_soc_history, rl_action_history, rl_reward_history, price_history = evaluate_on_fixed_windows(model,
                                                                                                                pv_forecast,
                                                                                                                price_forecast,
                                                                                                                start_indices)

    random_rewards, random_soc_history, random_action_history, random_reward_history, _ = evaluate_on_fixed_windows(
        'random', pv_forecast, price_forecast, start_indices)

    greedy_rewards, greedy_soc_history, greedy_action_history, greedy_reward_history, _ = evaluate_on_fixed_windows(
        'greedy', pv_forecast, price_forecast, start_indices)

    nothing_rewards, nothing_soc_history, nothing_action_history, nothing_reward_history, _ = evaluate_on_fixed_windows(
        'do_nothing', pv_forecast, price_forecast, start_indices)

    TIME_STEPS_PER_EPISODE = forecast_horizon
    TOTAL_EPISODES = nr_starts

    def split_history_into_episodes(history, time_steps_per_episode):
        return [history[i * time_steps_per_episode: (i + 1) * time_steps_per_episode]
                for i in range(TOTAL_EPISODES)]

    rl_soc_episodes = split_history_into_episodes(rl_soc_history, TIME_STEPS_PER_EPISODE)
    rl_action_episodes = split_history_into_episodes(rl_action_history, TIME_STEPS_PER_EPISODE)
    rl_reward_episodes = split_history_into_episodes(rl_reward_history, TIME_STEPS_PER_EPISODE)

    def create_episode_table(episodes, variable_name):
        rows = []
        for episode_idx, episode_data in enumerate(episodes):
            for step_idx, value in enumerate(episode_data):
                rows.append({
                    'Episode': episode_idx + 1,
                    'Time Step': step_idx + 1,
                    variable_name: value
                })
        return pd.DataFrame(rows)

    soc_df = create_episode_table(rl_soc_episodes, 'SOC (kWh)')
    action_df = create_episode_table(rl_action_episodes, 'Actions (Store, Discharge)')
    reward_df = create_episode_table(rl_reward_episodes, 'Reward')

    soc_df.to_csv(f"rl_soc_episodes_{version_prefix}.csv", index=False)
    action_df.to_csv(f"rl_action_episodes_{version_prefix}.csv", index=False)
    reward_df.to_csv(f"rl_reward_episodes_{version_prefix}.csv", index=False)

    def plot_soc_and_actions(soc_history, action_history, model_name):
        # Plot SOC and Actions
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(soc_history, label='State of Charge (kWh)', color='blue')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('SOC (kWh)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot([a[0] for a in action_history], label='Store Fraction', color='green', linestyle='--')
        ax2.plot([a[1] for a in action_history], label='Discharge (kW)', color='red', linestyle=':')
        ax2.set_ylabel('Actions', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        plt.title(f'{model_name} Agent Behavior Over Time')
        plt.legend(loc='upper left')
        plt.show()

    rl_avg, rl_std, rl_sum = np.mean(rl_rewards), np.std(rl_rewards), np.sum(rl_rewards)
    random_avg, random_std, random_sum = np.mean(random_rewards), np.std(random_rewards), np.sum(random_rewards)
    greedy_avg, greedy_std, greedy_sum = np.mean(greedy_rewards), np.std(greedy_rewards), np.sum(greedy_rewards)
    nothing_avg, nothing_std, nothing_sum = np.mean(nothing_rewards), np.std(nothing_rewards), np.sum(nothing_rewards)

    t_stat_rl_vs_greedy, p_value_rl_vs_greedy = stats.ttest_rel(rl_rewards, greedy_rewards)
    t_stat_rl_vs_nothing, p_value_rl_vs_nothing = stats.ttest_rel(rl_rewards, nothing_rewards)

    if plot_vis:
        plot_soc_and_actions(rl_soc_history, rl_action_history, "RL")

        print("\n" + "=" * 80)
        print("FINAL EVALUATION RESULTS (2026 Data - PAIRED TEST)")
        print("=" * 80)
        print(f"{'Strategy':<15} | {'Avg Reward (Eur)':<12} | {'Std Dev':<12} | {'Sum Rewards':<12} | {'Win Rate vs RL'}")
        print("-" * 80)
        print(f"{'RL Agent':<15} | {rl_avg:<12.4f} | {rl_std:<12.4f} | {rl_sum:<12.4f} | {'-'}")
        print(
            f"{'Random':<15} | {random_avg:<12.4f} | {random_std:<12.4f} | {random_sum:<12.4f} | {np.sum(random_rewards < rl_rewards) / len(rl_rewards) * 100:.1f}%")
        print(
            f"{'Greedy':<15} | {greedy_avg:<12.4f} | {greedy_std:<12.4f} | {greedy_sum:<12.4f} | {np.sum(greedy_rewards < rl_rewards) / len(rl_rewards) * 100:.1f}%")
        print(
            f"{'Do Nothing':<15} | {nothing_avg:<12.4f} | {nothing_std:<12.4f} | {nothing_sum:<12.4f} | {np.sum(nothing_rewards < rl_rewards) / len(rl_rewards) * 100:.1f}%")
        print("=" * 80)

        print("\n--- Statistical Significance (Paired t-test) ---")
        print(f"RL vs Greedy: p-value = {p_value_rl_vs_greedy:.4f}")
        if p_value_rl_vs_greedy < 0.05:
            print("  ✅ Statistically significant difference (p < 0.05)")
        else:
            print("  ⚠️  No statistically significant difference (p ≥ 0.05)")

        print(f"RL vs Do Nothing: p-value = {p_value_rl_vs_nothing:.4f}")
        if p_value_rl_vs_nothing < 0.05:
            print("  ✅ Statistically significant difference (p < 0.05)")
        else:
            print("  ⚠️  No statistically significant difference (p ≥ 0.05)")

        plt.figure(figsize=(12, 6))
        window_numbers = range(1, len(start_indices) + 1)
        plt.plot(window_numbers, rl_rewards, label='RL Agent', marker='o', markersize=3)
        plt.plot(window_numbers, greedy_rewards, label='Greedy', marker='s', markersize=3, alpha=0.7)
        plt.plot(window_numbers, nothing_rewards, label='Do Nothing', marker='^', markersize=3, alpha=0.5)
        plt.xlabel('Window Number (Same for all strategies)')
        plt.ylabel('Reward (EUR)')
        plt.title('Per-Window Performance Comparison (2026 Data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        window_diff = greedy_rewards - rl_rewards
        beating_indices = np.where(window_diff > 0.01)[0]  # Where Greedy wins by >0.01 EUR
        losing_indices = np.where(window_diff < 0.01)[0]  # Where Greedy loses by >0.01 EUR


        print(f"Greedy beats RL in {len(beating_indices)}/{len(window_diff)} windows")
        print(f"Greedy loses to RL in {len(losing_indices)}/{len(window_diff)} windows")

        from random import sample,seed
        seed(a=seed_val)
        random_windows = sample(range(nr_starts), 10)
        print(random_windows)

        for i in random_windows:
            start = start_indices[i]
            end = start + forecast_horizon
            fig, ax1 = plt.subplots(figsize=(12, 6))

            ax1.plot(rl_soc_history[start:end], color='blue', label='RL SOC')
            ax1.plot(greedy_soc_history[start:end], color='red', label='Greedy SOC')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('SOC (kWh)', color='white')
            ax1.tick_params(axis='y', labelcolor='white')
            plt.legend(loc='upper right')

            ax2 = ax1.twinx()
            ax2.plot(price_history[start:end], label='Price (Eur/kWh)', color='green', linestyle='--')
            ax2.set_ylabel('Price (Eur/kWh)', color='white')
            ax2.tick_params(axis='y', labelcolor='white')

            plt.title(f'Window {i} SOC vs Price Performance Comparison RL vs Greedy')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            plt.show()
