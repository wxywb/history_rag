from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.bridge.pydantic import Field
from typing import (
    Any,
    Optional,
    Sequence,
)
import openai

PROXY_API_KEY = "EMPTY"
PROXY_SERVER_URL = "http://x.x.x.x:20000/api/v1"
LLM_MODEL = "chatglm3-6b-32k"


class ProxyModel(LLM):
    model_name: str = Field(
        default=LLM_MODEL, description="The Proxy_model to use."
    )

    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    temperature: float = Field(
        default=0.1,
        description="The temperature to use for sampling.",
        gt=0,
    )

    api_base: str = Field(
        default=PROXY_SERVER_URL,
        description="The base URL for the Proxy_model API.",
    )


    api_key: str = Field(
        default=PROXY_API_KEY,
        description="The Proxy_model Key.",
    )

    def __init__(
            self,
            model_name: str = model_name,
            api_base: str = api_base,
            api_key: str = api_key,
            temperature: float = 0.1,
            max_tokens: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def call_with_prompt(self, prompt):
        client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        # 如果API一直有问题
        # 可以直接复制prompt到网页去问
        # print(prompt)

        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        if response is not None:
            return response.choices[0].message.content

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        answer = self.call_with_prompt(prompt)

        return CompletionResponse(
            text=answer,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=6000,
            num_output=self.max_tokens or -1,
            # is_chat_model=is_chat_model(model=self._get_model_name()),
            is_chat_model=False,
            is_function_calling_model=False,
            # is_function_calling_model=is_function_calling_model(
            #     model=self._get_model_name()
            # ),
            model_name=self.model_name,
        )

    # 下面是实现Interface必要的方法
    # 但这里用不到，所以都是pass
    @llm_completion_callback()
    async def astream_complete(self) -> CompletionResponseAsyncGen:
        pass

    async def _astream_chat(self) -> ChatResponseAsyncGen:
        pass

    @llm_chat_callback()
    async def astream_chat(self) -> ChatResponseAsyncGen:
        pass

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        pass

    @llm_chat_callback()
    def stream_chat(self) -> ChatResponseGen:
        pass

    @llm_completion_callback()
    def stream_complete(self) -> CompletionResponseGen:
        pass

    @llm_chat_callback()
    async def achat(self) -> ChatResponse:
        pass

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        pass
