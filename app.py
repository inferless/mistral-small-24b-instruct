from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import inferless
from pydantic import BaseModel, Field
from typing import Optional

@inferless.request
class RequestObjects(BaseModel):
        prompt: str = Field(default="Implement a function to check if a given number is a prime number.")
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 0.1
        repetition_penalty: Optional[float] = 1.18
        top_k: Optional[int] = 40
        max_tokens: Optional[int] = 256

@inferless.response
class ResponseObjects(BaseModel):
        generated_text: str = Field(default='Test output')

class InferlessPythonModel:
    def initialize(self):
        model_id = "mistralai/Mistral-Small-24B-Instruct-2501"
        self.llm = LLM(model=model_id,gpu_memory_utilization=0.9,max_model_len=5000)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def infer(self, request: RequestObjects) -> ResponseObjects:
        sampling_params = SamplingParams(temperature=request.temperature,top_p=request.top_p,
                                         repetition_penalty=request.repetition_penalty,
                                         top_k=request.top_k,max_tokens=request.max_tokens
                                        )
        input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": request.prompt}], tokenize=False)
        result = self.llm.generate(input_text, sampling_params)
        result_output = [output.outputs[0].text for output in result]
        
        generateObject = ResponseObjects(generated_text = result_output[0])        
        return generateObject

    def finalize(self):
        self.llm = None
