import secrets
import string
import openai
from abc import ABC
from typing import Optional, List
from langchain.llms.base import LLM
from models.loader import LoaderCheckPoint
from models.base import (RemoteRpcModel,
                         AnswerResult)

openai.api_type = "azure"
openai.api_base = "https://cog-3gp5kabpguwrg.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = ""


def clear_history(list):
    new = []
    for h in list:
        if h[0]:
            res = h[1].split("<details> <summary>", 1)
            new.append(h[0])
            new.append(res[0])
    return new


def generate_chat_message(history):
    message = []
    for i in history:
        if i[0]:
            ask = {"role": "assistant", "content": i[1]}
            user = {"role": "user", "content": i[0]}
            message.append(ask)
            message.append(user)
    return message


class Azure(RemoteRpcModel, LLM, ABC):

    ability_type = "chatGLM"  # 引擎类型
    engine_type = "chatGLM"  # 请求参数样例
    model_name: str = "chatglm-6b"
    temperature: float = 0.9
    top_p = 0.7
    checkPoint: LoaderCheckPoint = None
    history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "Azure"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    @property
    def _api_key(self) -> str:
        pass

    @property
    def _api_base_url(self) -> str:
        return self.api_base_url

    def set_api_key(self, api_key: str):
        pass

    def set_api_base_url(self, api_base_url: str):
        self.api_base_url = api_base_url

    def call_model_name(self, model_name):
        self.model_name = model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):
        answer_result = AnswerResult()
        print("Azure history: ", history)
        new_history = clear_history(history)
        print("Azure new_history: ", new_history)
        print("Azure content: ", prompt)

        messages = generate_chat_message(new_history)
        current_ask = {"role": "user", "content": prompt}
        messages.append(current_ask)
        print("Azure messages 参数如下:", messages)
        response = openai.ChatCompletion.create(
            engine="chat",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        # print("Azure API 返回如下:", response)
        res_text = response['choices'][0]['message']['content']
        if prompt:
            history += [[prompt, res_text]]
        answer_result.history = history
        answer_result.llm_output = {"answer": res_text}
        yield answer_result
