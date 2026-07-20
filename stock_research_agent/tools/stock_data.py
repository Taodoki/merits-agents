"""Stock data tools — delegates to multi-source data fetcher."""

from tools.data_sources import fetch_all

# Re-export for convenience
get_all_data = fetch_all
