"""Main clustering logic."""

import pandas as pd

from llm_clustering.config import get_settings
from llm_clustering.llm import get_llm_provider


class Clusterer:
    """Main clustering class for customer inquiries."""

    def __init__(self, llm_provider: str | None = None) -> None:
        """Initialize clusterer with LLM provider."""
        self.settings = get_settings()
        self.llm = get_llm_provider(llm_provider)

    def cluster_inquiries(
        self,
        inquiries: pd.DataFrame,
        text_column: str = "text",
        num_clusters: int | None = None,
    ) -> pd.DataFrame:
        """Cluster customer inquiries."""
        # TODO: Implement clustering logic
        raise NotImplementedError

    def describe_clusters(self, clustered_data: pd.DataFrame) -> pd.DataFrame:
        """Generate descriptions for each cluster."""
        # TODO: Implement cluster description
        raise NotImplementedError

