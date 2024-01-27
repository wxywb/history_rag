from typing import (
    Any,
    Optional,
    Sequence,
)

from llama_index.bridge.pydantic import Field
from llama_index.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.llm import LLM
from llama_index.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)

DEFAULT_MODEL = "qwen-turobo"
#export DASHSCOPE_API_KEY="YOUR_KEY"

import random
from http import HTTPStatus
import dashscope

class QwenUnofficial(LLM) :
    model: str = Field(
        default=DEFAULT_MODEL, description="The QWen model to use."
    )

    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None :

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    # Qwen 官方案例
    def call_with_prompt( self, prompt ):
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_max,
            prompt=prompt
        )

        # 如果API一直有问题
        # 可以直接复制prompt到网页去问
        # print( prompt )

        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        if response.status_code == HTTPStatus.OK:
            return response.output["text"]
        else:
            errMessage = (
                "通义模型API返回的错误: \n"
                "Error Code: " + response.code + "\n"
                "Error Message: " + response.message + "\n"
                "Request ID: " + response.request_id 
            )

            raise Exception( errMessage ) 

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        answer = self.call_with_prompt( prompt )

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
            model_name=self.model,
        )

    # 下面是实现Interface必要的方法
    # 但这里用不到，所以都是pass
    @llm_completion_callback()
    async def astream_complete() -> CompletionResponseAsyncGen:
        pass

    async def _astream_chat() -> ChatResponseAsyncGen:
        pass

    @llm_chat_callback()
    async def astream_chat() -> ChatResponseAsyncGen:
        pass

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        pass

    @llm_chat_callback()
    def stream_chat() -> ChatResponseGen:
        pass
    
    @llm_completion_callback()
    def stream_complete() -> CompletionResponseGen:
        pass

    @llm_chat_callback()
    async def achat() -> ChatResponse:
        pass
    
    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        pass