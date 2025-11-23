"""Examples of using llm_clustering library.

This file demonstrates various use cases:
1. Basic usage with default settings
2. Using custom LLM provider
3. Iterative processing with partial results
4. Re-fitting with existing cluster knowledge
5. Saving and loading clusters
6. Using business context for domain-specific clustering
"""

import pandas as pd
from pathlib import Path

# Import the library
from llm_clustering import (
    ClusteringPipeline,
    ClusteringResult,
    BaseLLMProvider,
    Settings,
)


# =============================================================================
# Example 1: Basic Usage with Default Settings
# =============================================================================
def example_1_basic_usage():
    """Simplest way to cluster text data."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Create sample data
    df = pd.DataFrame({
        "text": [
            "Не могу войти в аккаунт",
            "Забыл пароль, помогите восстановить",
            "Как изменить свой пароль?",
            "Товар не пришел уже неделю",
            "Когда будет доставка заказа?",
            "Трекинг номер не работает",
        ]
    })
    
    # Initialize pipeline with default settings
    pipeline = ClusteringPipeline()
    
    # Cluster the data
    result = pipeline.fit(df, text_column="text")
    
    # Display results
    print(f"\nProcessed {result.total_requests} requests")
    print(f"Assigned {result.assigned_requests} requests to clusters")
    print(f"Coverage: {result.coverage:.1f}%")
    print(f"\nFound {len(result.clusters)} clusters:")
    
    for cluster in result.clusters:
        print(f"  - {cluster.name} (count: {cluster.count})")
        print(f"    {cluster.summary}")
    
    # Access assignments
    print(f"\nAssignments shape: {result.assignments.shape}")
    print(result.assignments.head())


# =============================================================================
# Example 2: Custom LLM Provider
# =============================================================================
class CustomLLMProvider(BaseLLMProvider):
    """Example of custom LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Implement your custom LLM call here."""
        # This is a placeholder - implement your actual LLM call
        # For example, calling your own API:
        # response = requests.post(
        #     "https://your-llm-api.com/chat",
        #     json={"messages": messages, "temperature": temperature},
        #     headers={"Authorization": f"Bearer {self.api_key}"}
        # )
        # return response.json()["content"]
        
        raise NotImplementedError("Implement your LLM provider here")
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Optional: implement embeddings if needed."""
        raise NotImplementedError("Embeddings not implemented")
    
    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Optional: implement clustering if needed."""
        raise NotImplementedError("Clustering not implemented")
    
    def describe_cluster(self, texts: list[str]) -> str:
        """Optional: implement cluster description if needed."""
        raise NotImplementedError("Cluster description not implemented")


def example_2_custom_llm():
    """Use custom LLM provider."""
    print("=" * 80)
    print("Example 2: Custom LLM Provider")
    print("=" * 80)
    
    # Create your custom LLM provider
    custom_llm = CustomLLMProvider(
        api_key="your-api-key",
        model="your-model-name"
    )
    
    # Initialize pipeline with custom LLM
    pipeline = ClusteringPipeline(llm_provider=custom_llm)
    
    # Rest is the same as example 1
    df = pd.DataFrame({"text": ["Example text 1", "Example text 2"]})
    # result = pipeline.fit(df, text_column="text")
    
    print("Custom LLM provider initialized successfully")
    print("(Skipping actual clustering - implement your LLM first)")


# =============================================================================
# Example 3: Iterative Processing
# =============================================================================
def example_3_iterative_processing():
    """Process data in batches for better control."""
    print("=" * 80)
    print("Example 3: Iterative Processing")
    print("=" * 80)
    
    # Create larger dataset
    df = pd.DataFrame({
        "text": [f"User request number {i}" for i in range(100)]
    })
    
    # Initialize pipeline
    pipeline = ClusteringPipeline()
    
    # Process in batches of 20
    print(f"\nProcessing {len(df)} requests in batches...")
    
    for partial_result in pipeline.fit_partial(
        df,
        text_column="text",
        batch_size=20,
        start_from=0
    ):
        print(f"\nBatch {partial_result.batch_number}:")
        print(f"  Processed: {partial_result.processed_rows}/{partial_result.total_rows}")
        print(f"  New clusters discovered: {len(partial_result.new_clusters)}")
        
        if partial_result.new_clusters:
            for cluster in partial_result.new_clusters:
                print(f"    - {cluster.name}")
        
        # You can break early if needed
        # if partial_result.processed_rows >= 60:
        #     print("\nStopping early as requested")
        #     break
    
    print(f"\nFinal cluster count: {len(pipeline.get_clusters())}")


# =============================================================================
# Example 4: Re-fitting with Existing Knowledge
# =============================================================================
def example_4_refitting():
    """Re-cluster data using existing cluster knowledge."""
    print("=" * 80)
    print("Example 4: Re-fitting with Existing Knowledge")
    print("=" * 80)
    
    # First clustering run
    df1 = pd.DataFrame({
        "text": [
            "Не могу войти",
            "Забыл пароль",
            "Товар не пришел",
        ]
    })
    
    pipeline = ClusteringPipeline()
    result1 = pipeline.fit(df1, text_column="text")
    
    print(f"\nFirst run: found {len(result1.clusters)} clusters")
    print(f"Coverage: {result1.coverage:.1f}%")
    
    # New data arrives
    df2 = pd.DataFrame({
        "text": [
            "Как сбросить пароль?",
            "Проблема с входом в систему",
            "Где мой заказ?",
            "Возврат товара",
        ]
    })
    
    print("\nNew data arrived, re-clustering...")
    
    # Re-cluster with existing knowledge
    result2 = pipeline.refit(
        df2,
        previous_assignments=result1.assignments,
        text_column="text"
    )
    
    print(f"\nSecond run: {len(result2.clusters)} total clusters")
    print(f"Coverage: {result2.coverage:.1f}%")
    
    # The pipeline now has refined clusters based on both datasets
    all_clusters = pipeline.get_clusters()
    print(f"\nAll clusters (sorted by frequency):")
    for cluster in all_clusters:
        print(f"  - {cluster.name}: {cluster.count} requests")


# =============================================================================
# Example 5: Saving and Loading Clusters
# =============================================================================
def example_5_save_load_clusters():
    """Save clusters to file and load them later."""
    print("=" * 80)
    print("Example 5: Saving and Loading Clusters")
    print("=" * 80)
    
    # Create and fit pipeline
    df = pd.DataFrame({
        "text": ["Request 1", "Request 2", "Request 3"]
    })
    
    pipeline1 = ClusteringPipeline()
    result = pipeline1.fit(df, text_column="text")
    
    print(f"\nClustered data: {len(result.clusters)} clusters found")
    
    # Save clusters
    clusters_file = Path("ai_data/my_clusters.json")
    pipeline1.save_clusters(clusters_file)
    print(f"Clusters saved to: {clusters_file}")
    
    # Later... create new pipeline and load clusters
    pipeline2 = ClusteringPipeline()
    pipeline2.load_clusters(clusters_file)
    
    loaded_clusters = pipeline2.get_clusters()
    print(f"\nLoaded {len(loaded_clusters)} clusters from file")
    
    # Now you can use these clusters for new data
    new_df = pd.DataFrame({"text": ["New request 1", "New request 2"]})
    # new_result = pipeline2.fit(new_df, text_column="text")


# =============================================================================
# Example 6: Business Context
# =============================================================================
def example_6_business_context():
    """Use business context to guide clustering."""
    print("=" * 80)
    print("Example 6: Business Context")
    print("=" * 80)
    
    # Define business context
    business_context = """
    Контекст: Разметка обращений клиентов интернет-магазина.
    Цель: Разделить проблемы по сложности доработки бота.
    
    Категории сложности:
    - Простые: FAQ, типовые вопросы (можно автоматизировать)
    - Средние: требуют уточнения данных, проверки статуса
    - Сложные: нестандартные ситуации, требуют человека
    
    При кластеризации учитывай именно сложность автоматизации ответа.
    """
    
    df = pd.DataFrame({
        "text": [
            "Как оформить заказ?",
            "Где мой заказ номер 12345?",
            "Товар пришел сломанный, хочу вернуть деньги и написать жалобу",
            "Какие способы оплаты доступны?",
            "Можно ли отменить заказ?",
            "Курьер нагрубил и отказался поднять заказ на 5 этаж",
        ]
    })
    
    # Create pipeline with business context
    pipeline = ClusteringPipeline(business_context=business_context)
    
    # Cluster data - the LLM will use business context
    result = pipeline.fit(df, text_column="text")
    
    print(f"\nClustered with business context:")
    print(f"Found {len(result.clusters)} clusters")
    
    for cluster in result.clusters:
        print(f"\n  Cluster: {cluster.name}")
        print(f"  Summary: {cluster.summary}")
        print(f"  Count: {cluster.count}")


# =============================================================================
# Example 7: Complete Workflow
# =============================================================================
def example_7_complete_workflow():
    """Complete workflow with all features."""
    print("=" * 80)
    print("Example 7: Complete Workflow")
    print("=" * 80)
    
    # Setup
    business_context = "Разметка для улучшения чат-бота службы поддержки"
    custom_registry = Path("ai_data/custom_clusters.json")
    
    # Custom settings
    settings = Settings(
        clustering_batch_size=30,
        max_clusters_per_batch=5,
        default_temperature=0.1,
    )
    
    # Initialize pipeline
    pipeline = ClusteringPipeline(
        settings=settings,
        business_context=business_context,
        registry_path=custom_registry,
    )
    
    # Load data
    df = pd.read_csv("your_data.csv") if Path("your_data.csv").exists() else \
         pd.DataFrame({"text": [f"Sample request {i}" for i in range(50)]})
    
    print(f"\nLoaded {len(df)} requests")
    
    # Process iteratively
    all_assignments = []
    for partial in pipeline.fit_partial(df, text_column="text", batch_size=20):
        print(f"Batch {partial.batch_number}: {partial.processed_rows}/{partial.total_rows}")
        all_assignments.append(partial.assignments)
    
    # Combine all assignments
    final_assignments = pd.concat(all_assignments, ignore_index=True)
    
    # Get final clusters
    clusters = pipeline.get_clusters()
    print(f"\nFinal results:")
    print(f"  Total requests: {len(final_assignments)}")
    print(f"  Clusters: {len(clusters)}")
    print(f"  Registry: {custom_registry}")
    
    # Save everything
    final_assignments.to_csv("ai_data/final_assignments.csv", index=False)
    pipeline.save_clusters(Path("ai_data/final_clusters.json"))
    
    print("\nWorkflow completed successfully!")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LLM Clustering Library - Examples")
    print("=" * 80)
    
    # Run examples
    # Note: Some examples are commented out to avoid long execution
    # Uncomment the ones you want to try
    
    try:
        # example_1_basic_usage()
        # example_2_custom_llm()
        # example_3_iterative_processing()
        # example_4_refitting()
        # example_5_save_load_clusters()
        # example_6_business_context()
        # example_7_complete_workflow()
        
        print("\n" + "=" * 80)
        print("To run examples, uncomment them in the __main__ section")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running example: {e}")
        print("Make sure to configure your LLM provider and settings properly")

