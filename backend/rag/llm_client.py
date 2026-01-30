import logging
from typing import Optional, Dict
import os
from backend.config import LLM_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model_name: str = None, use_vllm: bool = False):
        self.model_name = model_name or LLM_MODEL
        self.use_vllm = use_vllm
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing LLM Client with model: {self.model_name}")
        
        if use_vllm:
            self._initialize_vllm()
        else:
            self._initialize_transformers()

    def _initialize_vllm(self):
        try:
            from vllm import LLM, SamplingParams
            
            logger.info("Initializing vLLM engine...")
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=2048,
                gpu_memory_utilization=0.8
            )
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                stop=["</s>", "<end_of_turn>"]
            )
            logger.info("vLLM engine initialized successfully")
            
        except ImportError:
            logger.warning("vLLM not available, falling back to transformers")
            self.use_vllm = False
            self._initialize_transformers()
        except Exception as e:
            logger.error(f"Error initializing vLLM: {str(e)}")
            logger.warning("Falling back to transformers")
            self.use_vllm = False
            self._initialize_transformers()

    def _initialize_transformers(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info("Loading model with transformers...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            if device == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "5GB"}
                )
                logger.info("Model loaded on GPU with float16 precision")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(device)
                logger.info("Model loaded on CPU")
            
            logger.info("Model loaded successfully with transformers")
            
        except Exception as e:
            logger.error(f"Error initializing transformers: {str(e)}")
            raise

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        try:
            if self.use_vllm:
                return self._generate_vllm(prompt, max_tokens, temperature)
            else:
                return self._generate_transformers(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    def _generate_vllm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stop=["</s>", "<end_of_turn>"]
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        logger.info(f"Generated {len(generated_text)} characters with vLLM")
        return generated_text.strip()

    def _generate_transformers(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        logger.info(f"Generated {len(generated_text)} characters with transformers")
        return generated_text

    def create_prompt(self, context: str, question: str) -> str:
        prompt = f"""You are a precise information assistant for UET (University of Engineering and Technology).

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the exact information from the CONTEXT above
- Quote specific details from the context when possible
- If the context doesn't contain the answer, say "The provided context does not contain this information."
- Do NOT make up names, numbers, or any other details
- Be concise and accurate

ANSWER:"""
        return prompt


if __name__ == "__main__":
    try:
        llm = LLMClient(use_vllm=False)
        
        test_context = """
        The Department of Computer Science offers undergraduate and graduate programs.
        Admission requirements include a minimum CGPA of 3.0 for undergraduate programs.
        The department has 20 faculty members including 5 professors.
        """
        
        test_question = "What are the admission requirements?"
        
        prompt = llm.create_prompt(test_context, test_question)
        print("Prompt created successfully")
        print(f"Prompt length: {len(prompt)} characters")
        
        print("\nGenerating response...")
        response = llm.generate(prompt, max_tokens=256)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"Error: {e}")
