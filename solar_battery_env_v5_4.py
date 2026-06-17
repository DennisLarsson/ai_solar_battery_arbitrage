import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SolarBatteryEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 0}

    def __init__(self, pv_forecast, price_forecast, battery_capacity=2.04, max_pv=1.0):
        super(SolarBatteryEnv, self).__init__()

        self.pv_forecast = np.array(pv_forecast)  # forecast of PV production in Kwh
        self.price_forecast = np.array(price_forecast) # forecast of spot price on Eur/Kwh

        # Validate lengths match
        if len(self.pv_forecast) != len(self.price_forecast):
            raise ValueError(
                f"PV forecast length ({len(self.pv_forecast)}) must match price forecast length ({len(self.price_forecast)})")

        self.forecast_horizon = len(self.pv_forecast)
        self.battery_capacity = battery_capacity # Kwh
        self.max_pv = max_pv # Kwh
        self.current_step = 0
        self.soc = 0

        # Action space: [store_frac, discharge]
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
        )

        # Observation space: [SOC] + [PV_forecast] + [Price_forecast]
        # Shape = 1 + forecast_horizon + forecast_horizon
        obs_shape = 6 + self.forecast_horizon + self.forecast_horizon

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Current values (unnormalized for raw signal)
        current_pv = self.pv_forecast[self.current_step - 1]
        current_price = self.price_forecast[self.current_step - 1]

        norm_current_pv = (current_pv - np.min(self.pv_forecast)) / (
                np.max(self.pv_forecast) - np.min(self.pv_forecast) + 1e-8)
        norm_current_price = (current_price - np.min(self.price_forecast)) / (
                np.max(self.price_forecast) - np.min(self.price_forecast) + 1e-8)

        # forecast (normalized for pattern recognition)
        norm_pv = (self.pv_forecast - np.min(self.pv_forecast)) / (
                np.max(self.pv_forecast) - np.min(self.pv_forecast) + 1e-8)
        norm_price = (self.price_forecast - np.min(self.price_forecast)) / (
                np.max(self.price_forecast) - np.min(self.price_forecast) + 1e-8)

        # Time progression
        time_ratio = self.current_step / self.forecast_horizon

        if time_ratio < 1:
            current_price_relative_to_future_min = current_price / (
                        np.min(self.price_forecast[self.current_step:]) + 1e-8)
            current_price_relative_to_future_max = current_price / (
                        np.max(self.price_forecast[self.current_step:]) + 1e-8)
        else:
            current_price_relative_to_future_min = 1
            current_price_relative_to_future_max = 1

        # SOC ratio
        soc_ratio = self.soc / self.battery_capacity

        obs = np.concatenate([
            [soc_ratio],
            [norm_current_pv],
            [norm_current_price],
            [current_price_relative_to_future_min],
            [current_price_relative_to_future_max],
            [time_ratio],
            norm_pv,
            norm_price
        ])

        return obs.astype(np.float32)

    def step(self, action):
        store_frac, discharge = action

        # Safety check for step index
        if self.current_step >= self.forecast_horizon:
            # Should ideally be handled by 'done' flag, but good for safety
            return self._get_obs(), 0.0, True, False, {}

        pv = self.pv_forecast[self.current_step] # Kwh production at current step
        price = self.price_forecast[self.current_step] # Eur/Kwh price at current step

        # Store and sell fractions
        store_pv = pv * store_frac # Kwh of PV production charged
        sell_pv = pv * (1 - store_frac) # Kwh of PV production directly sold

        # Battery charging
        max_store = min(store_pv, (self.battery_capacity - self.soc), self.max_pv)
        actual_store = max_store * 0.95
        actual_sell = sell_pv + (store_pv - max_store)
        self.soc = self.soc + actual_store

        # Battery discharging
        max_discharge = min((discharge * self.soc), self.max_pv)
        actual_discharge = max_discharge * 0.95
        self.soc = self.soc - max_discharge

        # Reward
        reward = (actual_sell + actual_discharge) * price

        self.current_step += 1
        done = self.current_step >= self.forecast_horizon

        return self._get_obs(), reward, done, False, {}