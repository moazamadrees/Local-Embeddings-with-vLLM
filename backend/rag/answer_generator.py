import logging
from typing import Dict, List, Optional
from backend.rag.retriever import Retriever
from backend.rag.llm_client import LLMClient
from backend.guardrail.scope_validator import ScopeValidator
from backend.config import GUARDRAIL_MESSAGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, use_vllm: bool = False):
        logger.info("Initializing AnswerGenerator")
        
        try:
            self.retriever = Retriever()
            self.llm_client = LLMClient(use_vllm=use_vllm)
            self.scope_validator = ScopeValidator()
            
            logger.info("AnswerGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AnswerGenerator: {str(e)}")
            raise

    def generate_answer(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> Dict:
        try:
            logger.info(f"Processing question: '{question}'")
            
            is_valid, guardrail_response = self.scope_validator.validate_and_respond(question)
            
            if not is_valid:
                logger.info("Question rejected by guardrail")
                return {
                    'answer': guardrail_response,
                    'citations': [],
                    'sources': [],
                    'metadata': {
                        'guardrail_triggered': True,
                        'question': question
                    }
                }
            
            logger.info("Question passed guardrail validation")
            
            context, retrieved_data = self.retriever.retrieve_and_format(question, top_k=top_k)
            
            if not context:
                logger.warning("No relevant context retrieved")
                return {
                    'answer': "I couldn't find relevant information in the UET documents to answer your question.",
                    'citations': [],
                    'sources': [],
                    'metadata': {
                        'guardrail_triggered': False,
                        'retrieval_count': 0,
                        'question': question
                    }
                }
            
            prompt = self.llm_client.create_prompt(context, question)
            
            logger.info("Generating answer with LLM...")
            answer = self.llm_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            citations = [doc['content'] for doc in retrieved_data]
            
            sources = []
            for doc in retrieved_data:
                meta = doc.get('metadata', {})
                dist = doc.get('distance', 0.0)
                sources.append({
                    'chunk_id': meta.get('chunk_id', 0),
                    'source': meta.get('source', 'unknown'),
                    'relevance_score': float(1 - dist) if dist else 0.0
                })
            
            result = {
                'answer': answer,
                'citations': citations,
                'sources': sources,
                'metadata': {
                    'guardrail_triggered': False,
                    'retrieval_count': len(citations),
                    'question': question,
                    'top_k': top_k
                }
            }
            
            logger.info(f"Answer generated successfully with {len(citations)} citations")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            return {
                'answer': "An error occurred while processing your question. Please try again.",
                'citations': [],
                'sources': [],
                'metadata': {
                    'error': str(e),
                    'question': question
                }
            }

    def generate_answer_batch(self, questions: List[str], **kwargs) -> List[Dict]:
        logger.info(f"Processing batch of {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            result = self.generate_answer(question, **kwargs)
            results.append(result)
        
        logger.info(f"Batch processing complete: {len(results)} answers generated")
        return results


if __name__ == "__main__":
    try:
        generator = AnswerGenerator(use_vllm=False)
        
        test_questions = [
            "What are the admission requirements for Computer Science?",
            "Who is the chairman of the Electrical Engineering department?",
            "What is the weather today?"
        ]
        
        print("Testing Answer Generator:")
        print("=" * 80)
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            print("-" * 80)
            
            result = generator.generate_answer(question, top_k=3, max_tokens=256)
            
            print(f"Answer: {result['answer']}")
            print(f"\nCitations: {len(result['citations'])}")
            
            if result['citations']:
                for i, citation in enumerate(result['citations'][:2], 1):
                    print(f"  [{i}] {citation[:150]}...")
            
            print("\n" + "=" * 80)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the preprocessing pipeline first!")
