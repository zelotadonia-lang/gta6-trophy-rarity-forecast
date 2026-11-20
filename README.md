# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zelotadonia-lang/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zelotadonia-lang/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Trophy Tide forecasts every GTA VI trophy's earn rate on Exophase one week after launch. Each rate is split into earners and players tracked, both fitted on the recent day of reads, because Exophase only refreshes its counts on its own schedule: between refreshes the page is frozen, at a refresh the tracked count jumps and every rate whose earners did not keep up falls. Where the history holds no refresh at all, the forecast prices the refresh itself as a two-scenario mixture instead of pretending the flat page will last. Python 3.13, standard library only, no network; deterministic, runs in under a second.

On the kit's practice cases it scores about 0.69 mean pinball loss against a refresh proxy, versus about 1.69 for the persistence baseline; on a synthetic multi-day final built from the rules' own mechanics (refresh steps, saturating tracked growth, per-trophy progress curves) it scores about 0.35 versus 0.47 for the baseline.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=yk7ez00m&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
