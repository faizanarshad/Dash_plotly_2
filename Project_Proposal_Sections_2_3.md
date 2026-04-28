## 2. Why is this question interesting or important?

Public policy shapes everyday life. Decisions about social protection, healthcare, employment support, and family benefits influence not only economic outcomes but also how people experience their lives. That is why we focus on life satisfaction as a central outcome.

Across the world, governments allocate large shares of national income to social safety nets. Yet public debate often emphasizes the cost of these programs rather than their human impact. Subjective wellbeing offers a valuable policy lens because it captures lived experience beyond traditional macroeconomic indicators such as GDP.

Our project addresses this gap by building an exploratory dashboard to identify patterns, trends, and outliers, including countries that perform better than expected given their income level. We also examine whether safety nets may matter more for vulnerable groups, such as lower-income or unemployed populations.

In addition, we explore how social media usage may interact with policy context and wellbeing, since social media is increasingly recognized as a factor influencing behavior, perception, and mental wellbeing.

## 3. What data will you be using, and why are these datasets suitable for your focus area?

We use the World Happiness Report alongside multiple OECD datasets and selected World Bank indicators. These sources are well-suited to our research question because they can be linked on common keys, primarily country and year, enabling integrated analysis of policy, socioeconomic context, and wellbeing outcomes.

These datasets provide enough breadth and depth for methods such as:

- correlation analysis
- regression modeling
- clustering and comparative profiling

They are appropriate for our focus area because they allow us to compare happiness levels across countries and regions while testing potential explanatory factors such as GDP per capita, inequality, labor market conditions, corruption perceptions, and social support.

### Dataset selection and scope

The listed datasets represent an initial proposal-phase inventory. Final inclusion will be refined iteratively based on data availability, join compatibility, and relevance to the guiding question. Not every listed source is guaranteed to be used in the final dashboard.

### Data architecture and expected size

All four datasets can be integrated using a common country key and year where available, with the World Happiness Report as the base table. The recommended merge order is:

- Start with World Happiness Report as the base (`country`, `year`, `happiness_score`).
- Left join OECD Social Protection (`SOCX_AGG`) on `country_iso3 + year` to add `social_spending_pct_gdp`.
- Left join World Bank WDI on `country_iso3 + year` to add GDP, Gini, and unemployment controls.
- Left join OECD Better Life Index (BLI) on `country_iso3` (country-only in the current extract) to add wellbeing-dimension indicators for Viz 4.

Country-key standardization is required before merging. WHR uses country names, while OECD and WDI use ISO-style country codes and different naming conventions. A lookup table should be created during cleaning to map WHR country names to a consistent `country_iso3` key.

The resulting integrated dataset currently contains 1,440 country-year observations for 2015-2023 after harmonizing and merging the four selected sources.
