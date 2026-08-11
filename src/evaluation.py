"""Evaluation module using RAGAS metrics."""
import logging
from typing import List, Dict, Any

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    Dataset = None
    evaluate = None

from src.config import settings

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluates RAG system performance using RAGAS."""

    def __init__(self, llm_model: str = "gpt-3.5-turbo", embeddings_model: str = "text-embedding-ada-002"):
        """Initialize evaluator with models for evaluation."""
        if evaluate is None:
            raise ImportError("ragas and datasets are required for evaluation. Run: pip install ragas datasets")
            
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY is not set. RAGAS evaluation strongly recommends using OpenAI models for evaluation metrics.")
            
        # RAGAS natively works best with OpenAI for evaluating. 
        # Even if the RAG system uses Ollama, we typically use OpenAI as the judge.
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        # In a full implementation, you can pass custom LLMs/Embeddings to the evaluate function.
        logger.info("Initialized RAG Evaluator.")

    def run_evaluation(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run evaluation on a test dataset.
        
        Args:
            test_dataset: List of dicts with keys:
                - question (str)
                - answer (str) - The generated answer
                - contexts (List[str]) - The retrieved contexts
                - ground_truth (str) - The expected correct answer
                
        Returns:
            Dict containing the evaluation scores.
        """
        logger.info(f"Running evaluation on {len(test_dataset)} samples.")
        
        # Prepare data in the format expected by RAGAS / HuggingFace Datasets
        data = {
            "question": [item["question"] for item in test_dataset],
            "answer": [item["answer"] for item in test_dataset],
            "contexts": [item["contexts"] for item in test_dataset],
            "ground_truth": [item["ground_truth"] for item in test_dataset],
        }
        
        hf_dataset = Dataset.from_dict(data)
        
        try:
            # Note: For evaluation, RAGAS will use OpenAI models by default if OPENAI_API_KEY is set.
            result = evaluate(
                dataset=hf_dataset,
                metrics=self.metrics,
                # To use local models for eval, we would need to override llm and embeddings here.
            )
            
            # Convert to dict
            report = {
                "overall_scores": {
                    "faithfulness": result.get("faithfulness", 0.0),
                    "answer_relevancy": result.get("answer_relevancy", 0.0),
                    "context_precision": result.get("context_precision", 0.0),
                    "context_recall": result.get("context_recall", 0.0),
                },
                "raw_result": result.to_pandas().to_dict(orient="records")
            }
            logger.info("Evaluation complete.")
            return report
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"error": str(e)}
