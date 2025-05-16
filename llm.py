from llama_cpp import Llama
from openai import OpenAI
from loguru import logger

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
        if api_key:
            logger.debug(f"Initializing OpenAI client with base_url: {base_url}, model: {model}")
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            logger.debug("Initializing local Llama model")
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang
        # 设置 extra_body 参数，用于特定模型（如 Qwen/Qwen3-235B-A22B）的优化
        self.extra_body = {
            "enable_thinking": False  # 默认禁用思考功能，可根据需要调整
            # "thinking_budget": 4096  # 可选参数，控制思考的 token 数量
        }

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            logger.debug(f"Generating content with model: {self.model}")
            try:
                # 根据模型名称动态调整请求参数
                if self.model == "Qwen/Qwen3-235B-A22B":
                    logger.debug("Using optimized request scheme for Qwen/Qwen3-235B-A22B")
                    response = self.llm.chat.completions.create(
                        messages=messages,
                        temperature=0,
                        model=self.model,
                        stream=False,  # 显式设置为非流式输出
                        extra_body=self.extra_body  # 传递 extra_body 参数
                    )
                else:
                    logger.debug("Using default request scheme for other models")
                    response = self.llm.chat.completions.create(
                        messages=messages,
                        temperature=0,
                        model=self.model
                    )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"API error with model {self.model}: {str(e)}")
                raise
        else:
            response = self.llm.create_chat_completion(messages=messages, temperature=0)
            return response["choices"][0]["message"]["content"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    logger.debug(f"Setting global LLM with model: {model}, base_url: {base_url}")
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
