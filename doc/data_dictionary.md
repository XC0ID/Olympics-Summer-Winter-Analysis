# Data dictionary

## SummerSD.csv / WinterSD.csv
| Column  | Type   | Description                     |
|---------|--------|---------------------------------|
| Year    | int    | Olympic year                    |
| City    | string | Host city                       |
| Sport   | string | Sport name                      |
| Country | string | Country name                    |
| Athlete | string | Athlete full name               |
| Medal   | string | Gold / Silver / Bronze          |
| Season  | string | Summer or Winter (auto-added)   |

## CountriesSD.csv
| Column        | Type   | Description              |
|---------------|--------|--------------------------|
| Country       | string | Country name             |
| Code          | string | 3-letter country code    |
| Population    | float  | Total population         |
| GDP per Capita| float  | GDP per capita USD       |

## data/processed/features.csv
| Column        | Type  | Description                        |
|---------------|-------|------------------------------------|
| TotalMedals   | int   | Total medals won that year         |
| GoldMedals    | int   | Gold medals won                    |
| SilverMedals  | int   | Silver medals won                  |
| BronzeMedals  | int   | Bronze medals won                  |
| WeightedScore | int   | Gold×3 + Silver×2 + Bronze×1       |
| PrevMedals    | float | Medals won in previous games       |
| RollingMean3  | float | Rolling average last 3 games       |
| MedalDelta    | float | Change from previous games         |
| IsSummer      | int   | 1 = Summer, 0 = Winter             |