from typing import (
    Any,
    Optional,
    Sequence,
)

from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import LLM
from llama_index.core.llms import ChatMessage, ChatResponse, ChatResponseAsyncGen, ChatResponseGen, CompletionResponse, CompletionResponseAsyncGen, CompletionResponseGen, LLMMetadata

DEFAULT_MODEL = "gemini-pro"

import google.generativeai as genai
import os

class Gemini(LLM):
    model_name: str = Field(
        default=DEFAULT_MODEL, description="The Gemini model to use."
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
        default="generativelanguage.googleapis.com",
        description="The base URL for the Gemini API.",
    )

    def __init__(
            self,
            model_name: str = DEFAULT_MODEL,
            temperature: float = 0.1,
            max_tokens: Optional[int] = None,
            api_base: str = "generativelanguage.googleapis.com",
            **kwargs: Any,
    ) -> None:
        if model_name.find("models/") == -1:
            model_name = f"models/{model_name}"

        if os.getenv("GOOGLE_API_BASE") is not None:
            api_base = os.getenv("GOOGLE_API_BASE")

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            **kwargs,
        )


    def call_with_prompt(self, prompt):
        # export GOOGLE_API_KEY="YOUR_KEY"
        # export GOOGLE_API_BASE="generativelanguage.googleapis.com"
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"),
                        client_options={
                            "api_endpoint": self.api_base
                        },
                        transport='rest')
        model = genai.GenerativeModel(self.model_name,
                                      generation_config=genai.GenerationConfig(
                                          temperature=self.temperature,
                                          max_output_tokens=self.max_tokens
                                      ))
        response = model.generate_content(prompt)

        # 如果API一直有问题
        # 可以直接复制prompt到网页去问
        # print(prompt)

        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        if response is not None:
            return response.text

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
    async def astream_complete(self) -> CompletionResponseAsyncGen:
        pass

    async def _astream_chat(self) -> ChatResponseAsyncGen:
        pass

    async def astream_chat(self) -> ChatResponseAsyncGen:
        pass

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        pass

    def stream_chat(self) -> ChatResponseGen:
        pass

    def stream_complete(self) -> CompletionResponseGen:
        pass

    async def achat(self) -> ChatResponse:
        pass

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        pass
