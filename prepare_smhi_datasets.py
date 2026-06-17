import pandas as pd

def time_convert_spotprices(df, timedelta = None):
    df['time'] = pd.to_datetime(df['time'], utc=True)
    if timedelta:
        df['time'] = df['time'] - pd.Timedelta(hours=timedelta)
    df['time'] = df['time'].dt.tz_convert('Europe/Stockholm')
    return df

def quarterly_to_hourly_price(df, timedelta=None):
    df_td = time_convert_spotprices(df, timedelta)
    df_td_hourly = df_td.set_index('time').resample('h').mean().reset_index()
    return df_td_hourly

def read_smhi_predicted_pv_output(smhi_predicted_pv_output):
    smhi_pv_pred = pd.read_csv(smhi_predicted_pv_output)
    smhi_pv_pred.rename(columns={'predicted_pv_output': 'P'}, inplace=True)
    smhi_pv_pred_td = time_convert_spotprices(smhi_pv_pred, 1)
    return smhi_pv_pred_td

def read_energy_chart_spotprices(energy_chart_spotprices):
    spotprices = pd.read_csv(energy_chart_spotprices, header=2)
    spotprices.columns = ['time', 'price']
    spotprices_hourly = quarterly_to_hourly_price(spotprices)
    min_rows = spotprices_hourly.count().min()
    spotprices_hourly_tr = spotprices_hourly.iloc[:min_rows]
    return spotprices_hourly_tr

def are_datetimes_identical(entry1, entry2, time_col='time'):
    # Convert to datetime if not already
    dt1 = pd.to_datetime(entry1[time_col], utc=True)
    dt2 = pd.to_datetime(entry2[time_col], utc=True)

    # Compare timestamps (including timezone)
    return dt1 == dt2
