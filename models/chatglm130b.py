import secrets
import string
from abc import ABC
from typing import Optional, List
from langchain.llms.base import LLM
from models.loader import LoaderCheckPoint
from models.base import (RemoteRpcModel,
                         AnswerResult)
from wudao.api_request import executeEngine, getToken


def generate_random_string(length):
    # 生成包含大小写字母和数字的序列
    characters = string.ascii_letters + string.digits
    # 随机选择 length 个字符
    return ''.join(secrets.choice(characters) for i in range(length))
def isBlank(myString):
    return not (myString and myString.strip())


def clear_history(list):
    new = []
    for h in list:
        if h[0]:
            res = h[1].split("<details> <summary>", 1)
            new.append(h[0])
            new.append(res[0])
    return new

class ChatGLM130B(RemoteRpcModel, LLM, ABC):
    # 接口 API KEY
    API_KEY = ""
    # 公钥
    PUBLIC_KEY = ""
    ability_type = "chatGLM"  # 引擎类型
    engine_type = "chatGLM"  # 请求参数样例
    # token_result = getToken(API_KEY, PUBLIC_KEY)
    # if token_result and token_result["code"] == 200:
    #     token = token_result["data"]
    # else:
    #     print("获取 token 失败，请检查 API_KEY 和 PUBLIC_KEY")
    token = 'eyJhbGciOiJIUzUxMiJ9.eyJ1c2VyX3R5cGUiOiJTRVJWSUNFIiwidXNlcl9pZCI6MTMzOTg3LCJhcGlfa2V5IjoiNDNjY2YxZDQxNjA4NDQ5N2E4ZTY1YWU0Njc4OGExZmEiLCJ1c2VyX2tleSI6IjBjYmJlMzkyLTUwZDMtNDFjYi1iMzk3LTRjNTE1OThiYzcxNiIsImN1c3RvbWVyX2lkIjoiNzYyNTU0NTkzMzkwODk0NDU1OSIsInVzZXJuYW1lIjoiMTg2NTg4NjYzMjgifQ.jxIw20jUxwV65DUmWrfmKMuKX-5kS70YbQAYWT8x4trveEt94GWtRCzwZ0YF9yVXqyVY3f44VnzDWMucCzK65w'
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
        return "ChatGLM130B"

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
        requestTaskNo = generate_random_string(20)
        answer_result = AnswerResult()
        print("generatorAnswer history: ", history)
        new_history = clear_history(history)
        data = {
            "top_p": self.top_p,
            "temperature": self.temperature,
            "prompt": prompt,
            "requestTaskNo": requestTaskNo,
            "history": new_history

        }
        print("generatorAnswer new_history: ", new_history)
        print("130B API 参数如下:", data )
        resp = executeEngine(self.ability_type, self.engine_type, self.token, data)
        print("130B API 返回如下:", resp )
        if prompt:
            history += [[prompt, resp['data']['outputText']]]
      
        answer_result.history = history
        answer_result.llm_output = {"answer": resp['data']['outputText']}
        yield answer_result
