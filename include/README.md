# tripadvisor-foodbot's submodules

This directory contains submodules of tripadvisor-foodbot.

- `tripadvisor-crawler`: A crawler that scrapes data from `tripadvisor` using `httpx` and `beautifulsoup4`.
- `tripadvisor-dbt-transformer`: A `dbt` project that transforms raw data from tripadvisor into a data warehouse.
- `tripadvisor-absa-llm`: A project that uses LLM to rate reviews from tripadvisor based on aspect.

Theses modules are used as a core part of tripadvisor-foodbot and are producing data for `tripadvisor-foodbot`.