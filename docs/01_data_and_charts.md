# BikePulse: understanding the data and EDA

## What question are we answering?

How many rentals will the entire Capital Bikeshare system record in the next hour, given calendar information and recently observed demand? This is system-wide demand forecasting. It does not tell us how many bikes are available at a particular station.

The source is the real [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), DOI 10.24432/C5W894, attributed to Fanaee-T and Gama under CC BY 4.0. It describes Washington, D.C. Capital Bikeshare observations from January 2011 through December 2012. The hourly CSV has 17,379 rows and 17 original columns. One row describes one recorded hour, not one customer or trip. Our loader creates a timestamp from date plus hour and uses it as the index.

## Every original column

| Column | Meaning | Role in this project |
|---|---|---|
| `instant` | Record identifier | Excluded; not a meaningful demand input |
| `dteday` | Date | Builds the timestamp; calendar features are derived from it |
| `season` | 1 spring, 2 summer, 3 fall, 4 winter, according to the source coding | EDA context; excluded from the current model |
| `yr` | 0 for 2011, 1 for 2012 | EDA context; excluded from the current model |
| `mnth` | Month, 1–12 | Model month is derived from timestamp |
| `hr` | Hour, 0–23 | Model hour is derived from timestamp |
| `holiday` | Source holiday flag | Included; known calendar information |
| `weekday` | Source day-of-week code | Our model derives its own code: Monday=0 through Sunday=6 |
| `workingday` | 1 when neither weekend nor holiday | Excluded; derivable from weekday and holiday |
| `weathersit` | Weather category: 1 clear/few clouds; 2 mist/cloudy; 3 light rain/snow; 4 severe weather | EDA only; observed future-hour weather is unavailable at forecast time |
| `temp` | Normalized temperature; source divides Celsius temperature by 41 | EDA only in this version |
| `atemp` | Normalized feels-like temperature; source divides by 50 | EDA only in this version |
| `hum` | Relative humidity divided by 100 | EDA only in this version |
| `windspeed` | Wind speed divided by 67, per source documentation | EDA only in this version |
| `casual` | Rentals by casual users during this hour | Excluded: part of the target |
| `registered` | Rentals by registered users during this hour | Excluded: part of the target |
| `cnt` | Total rentals, `casual + registered` | Prediction target |

A dataset column is not automatically a model feature. The model receives just seven inputs: hour, weekday, month, holiday, and demand 1, 24, and 168 clock hours earlier. The latter three are engineered features rather than original columns. Giving it `casual` and `registered` would reveal the answer and make evaluation misleading.

## Why no null values does not mean complete data

`isna().sum()` checks whether recorded cells are empty. All original columns have zero null cells. That means the existing rows are filled in; it does not mean every hour has a row, every measurement is accurate, or there are no outliers.

January 2011 through December 2012 contains 17,544 clock-hour slots. We observe 17,379: **165 hourly timestamps are absent**. An absent row is different from a row containing a null value. We cannot safely assume those absent hours had zero rentals.

The loader checks unique timestamps, nonnegative counts, and that casual plus registered rentals equals the total. The feature builder looks up exact timestamp offsets rather than shifting row positions. If 03:00 is absent, the preceding-hour feature at 04:00 is missing; it must not silently use 02:00.

After requiring all three lags, 482 observations are excluded and 16,897 remain. Some exclusions occur at the beginning, where a full week of history is unavailable, and others occur around missing hours. Scores describe eligible hours only. This selection can affect representativeness.

## Reading the daily-demand chart

The horizontal axis is date. The vertical axis is the sum of recorded rentals for that date. Each point aggregates hourly `cnt` values into one day.

Look for seasonal rises and falls, recurring short-term fluctuations, unusual drops or peaks, and changing demand between years. Recorded annual totals are 1,243,103 rentals in 2011 and 2,049,576 in 2012. Those are recorded totals, not estimates corrected for absent hours.

This shows substantial growth in recorded demand, but it does not prove why demand grew. System expansion, usage changes, and weather could contribute; this dataset alone does not establish their causal effects. Daily fluctuations also should not automatically be called errors or outliers to remove.

A daily total can be incomplete if that date has missing hourly observations. We should examine hourly coverage before interpreting an unusually low day. The chart is useful for seeing changing conditions that can make forecasting harder.

## Reading the hourly commuting-pattern chart

The chart shows **commuting patterns**, not computing power. The horizontal axis is hour of day, 0–23. The vertical axis is mean recorded rentals per hour. The two lines group Monday–Friday versus Saturday–Sunday; public holidays are not removed from the weekday group.

In the executed data:

- Weekday averages are approximately 465 rentals at 08:00 and 516 at 17:00.
- Weekend averages are approximately 99 at 08:00 and 380 at 13:00.

Weekday morning and late-afternoon peaks are consistent with commuting. The weekend curve is more concentrated around midday, consistent with a different usage pattern. These are plausible interpretations, not direct evidence of riders' intentions.

This explains why both hour and day of week are useful model inputs: an 08:00 prediction on Monday should differ from an 08:00 prediction on Sunday. These averages combine both years and seasons. They are not forecasts or guaranteed values for a particular day, and missing observations can affect the averages.

## Can 2011–2012 data predict demand today?

It can train a model, demonstrate a sound workflow, and support historical evaluation. We have not established that it predicts present-day demand reliably. The system, station network, user behavior, pricing, commuting patterns, and demand levels may have changed. This is called distribution shift: the relationship learned from old data may no longer describe the current situation.

The current app deliberately limits its replay to eligible October–December 2012 hours. It does not offer today's forecast. Its lag inputs come from that historical dataset.

For a useful present-day version we need recent real observations, the same clearly defined prediction target, reliable hourly aggregation, current lag inputs, current holiday information, and evaluation on recent later dates. Then we must retrain and monitor errors. If adding weather, use forecasts available at prediction time rather than weather observed afterward. Station availability is a separate problem requiring inventory snapshots.

Old data is appropriate for the first portfolio release if we label its scope honestly. A recent-data extension is the next research phase, not a capability this release already has.
