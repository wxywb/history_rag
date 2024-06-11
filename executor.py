import argparse
import logging
import sys
import re
import os
import argparse

import requests
from pathlib import Path
from urllib.parse import urlparse

from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core import set_global_service_context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import FlatReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser

from llama_index.core import ChatPromptTemplate, PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.indices.managed.zilliz import ZillizCloudPipelineIndex
from llama_index.core import QueryBundle
from llama_index.core.schema import BaseNode, ImageNode, MetadataMode

from custom.history_sentence_window import HistorySentenceWindowNodeParser
from custom.llms.QwenLLM import QwenUnofficial
from custom.llms.GeminiLLM import Gemini
from custom.llms.proxy_model import ProxyModel
from pymilvus import MilvusClient

QA_PROMPT_TMPL_STR = (
    "请你仔细阅读相关内容，结合历史资料进行回答,每一条史资料使用'出处：《书名》原文内容'的形式标注 (如果回答请清晰无误地引用原文,先给出回答，再贴上对应的原文，使用《书名》[]对原文进行标识),，如果发现资料无法得到答案，就回答不知道 \n"
    "搜索的相关历史资料如下所示.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "问题: {query_str}\n"
    "答案: "
)

QA_SYSTEM_PROMPT = "你是一个严谨的历史知识问答智能体，你会仔细阅读历史材料并给出准确的回答,你的回答都会非常准确，因为你在回答的之后，使用在《书名》[]内给出原文用来支撑你回答的证据.并且你会在开头说明原文是否有回答所需的知识"

REFINE_PROMPT_TMPL_STR = ( 
    "你是一个历史知识回答修正机器人，你严格按以下方式工作"
    "1.只有原答案为不知道时才进行修正,否则输出原答案的内容\n"
    "2.修正的时候为了体现你的精准和客观，你非常喜欢使用《书名》[]将原文展示出来.\n"
    "3.如果感到疑惑的时候，就用原答案的内容回答。"
    "新的知识: {context_msg}\n"
    "问题: {query_str}\n"
    "原答案: {existing_answer}\n"
    "新答案: "
)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_github_folder_url(url):
    return url.startswith('https://raw.githubusercontent.com/') and '.' not in os.path.basename(url)


def get_branch_head_sha(owner, repo, branch):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{branch}"
    response = requests.get(url)
    data = response.json()
    sha = data['object']['sha']
    return sha

def get_github_repo_contents(repo_url):
    # repo_url example: https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/
    repo_owner = repo_url.split('/')[3]
    repo_name = repo_url.split('/')[4]
    branch = repo_url.split('/')[5]
    folder_path = '/'.join(repo_url.split('/')[6:])
    sha = get_branch_head_sha(repo_owner, repo_name, branch)
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{sha}?recursive=1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            raw_urls = []
            for file in data['tree']:
                if file['path'].startswith(folder_path) and file['path'].endswith('.txt'):
                    raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
                    raw_urls.append(raw_url)
            return raw_urls
        else:
            print(f"Failed to fetch contents. Status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to fetch contents. Error: {str(e)}")
    return []

class Executor:
    def __init__(self, model):
        pass

    def build_index(self, path, overwrite):
        pass

    def build_query_engine(self):
        pass
     
    def delete_file(self, path):
        pass
    
    def query(self, question):
        pass
 

class MilvusExecutor(Executor):
    def __init__(self, config):
        self.index = None
        self.query_engine = None
        self.config = config
        self.node_parser = HistorySentenceWindowNodeParser.from_defaults(
            sentence_splitter=lambda text: re.findall("[^,.;。？！]+[,.;。？！]?", text),
            window_size=config.milvus.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",)

        embed_model = HuggingFaceEmbedding(model_name=config.embedding.name)

        # 使用Qwen 通义千问模型
        if config.llm.name.find("qwen") != -1:
            llm = QwenUnofficial(temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)
        elif config.llm.name.find("gemini") != -1:
            llm = Gemini(temperature=config.llm.temperature, model_name=config.llm.name, max_tokens=2048)
        elif 'proxy_model' in config.llm:
            llm = ProxyModel(model_name=config.llm.name, api_base=config.llm.api_base, api_key=config.llm.api_key,
                             temperature=config.llm.temperature,  max_tokens=2048)
            print(f"使用{config.llm.name},PROXY_SERVER_URL为{config.llm.api_base},PROXY_API_KEY为{config.llm.api_key}")
        else:
            api_base = None
            if 'api_base' in config.llm:
                api_base = config.llm.api_base
            llm = OpenAI(api_base = api_base, temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)

        Settings.llm = llm
        Settings.embed_model = embed_model
        rerank_k = config.milvus.rerank_topk
        self.rerank_postprocessor = SentenceTransformerRerank(
            model=config.rerank.name, top_n=rerank_k)
        self._milvus_client = None
        self._debug = False
        
    def set_debug(self, mode):
        self._debug = mode

    def build_index(self, path, overwrite):
        config = self.config
        uri = f"http://{config.milvus.host}:{config.milvus.port}",
        vector_store = MilvusVectorStore(
            uri = f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name = config.milvus.collection_name,
            overwrite=overwrite,
            dim=config.embedding.dim)
        self._milvus_client = vector_store._milvusclient
         
        if path.endswith('.txt'):
            if os.path.exists(path) is False:
                print(f'(rag) 没有找到文件{path}')
                return
            else:
                documents = FlatReader().load_data(Path(path))
                documents[0].metadata['file_name'] = documents[0].metadata['filename'] 
        elif os.path.isfile(path):           
            print('(rag) 目前仅支持txt文件')
        elif os.path.isdir(path):
            if os.path.exists(path) is False:
                print(f'(rag) 没有找到目录{path}')
                return
            else:
                documents = SimpleDirectoryReader(path).load_data()
        else:
            return

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

    def _get_index(self):
        config = self.config
        vector_store = MilvusVectorStore(
            uri = f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name = config.milvus.collection_name,
            dim=config.embedding.dim,
            )
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self._milvus_client = vector_store._milvusclient

    def build_query_engine(self):
        config = self.config
        if self.index is None:
            self._get_index()
        self.query_engine = self.index.as_query_engine(node_postprocessors=[
            self.rerank_postprocessor,
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ])
        self.query_engine._retriever.similarity_top_k=config.milvus.retrieve_topk

        message_templates = [
            ChatMessage(content=QA_SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=QA_PROMPT_TMPL_STR,
                role=MessageRole.USER,
            ),
        ]
        chat_template = ChatPromptTemplate(message_templates=message_templates)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": chat_template}
        )
        self.query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = REFINE_PROMPT_TMPL_STR

    def delete_file(self, path):
        config = self.config
        if self._milvus_client is None:
            self._get_index()
        num_entities_prev = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        res = self._milvus_client.delete(collection_name=config.milvus.collection_name, filter=f"file_name=='{path}'")
        num_entities = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        print(f'(rag) 现有{num_entities}条，删除{num_entities_prev - num_entities}条数据')
    
    def query(self, question):
        if self.index is None:
            self._get_index()
        if question.endswith('?') or question.endswith('？'):
            question = question[:-1]
        if self._debug is True:
            contexts = self.query_engine.retrieve(QueryBundle(question))
            for i, context in enumerate(contexts): 
                print(f'{question}', i)
                content = context.node.get_content(metadata_mode=MetadataMode.LLM)
                print(content)
            print('-------------------------------------------------------参考资料---------------------------------------------------------')
        response = self.query_engine.query(question)
        return response

class PipelineExecutor(Executor):
    def __init__(self, config):
        self.ZILLIZ_CLUSTER_ID = os.getenv("ZILLIZ_CLUSTER_ID")
        self.ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
        self.ZILLIZ_PROJECT_ID = os.getenv("ZILLIZ_PROJECT_ID") 
        self.ZILLIZ_CLUSTER_ENDPOINT = f"https://{self.ZILLIZ_CLUSTER_ID}.api.gcp-us-west1.zillizcloud.com"
    
        self.config = config
        if len(self.ZILLIZ_CLUSTER_ID) == 0:
            print('ZILLIZ_CLUSTER_ID 参数为空')
            exit()

        if len(self.ZILLIZ_TOKEN) == 0:
            print('ZILLIZ_TOKEN 参数为空')
            exit()
        
        self.config = config
        self._debug = False

        if config.llm.name.find("qwen") != -1:
            llm = QwenUnofficial(temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)
        elif config.llm.name.find("gemini") != -1:
            llm = Gemini(model_name=config.llm.name, temperature=config.llm.temperature, max_tokens=2048)
        else:
            api_base = None
            if 'api_base' in config.llm:
                api_base = config.llm.api_base
            llm = OpenAI(api_base = api_base, temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)

        Settings.llm = llm
        self._initialize_pipeline()


    def set_debug(self, mode):
        self._debug = mode

    def _initialize_pipeline(self):
        config = self.config
        try:
            pipeline_ids = self._list_pipeline_ids()
            self.pipeline_ids = pipeline_ids
            if len(pipeline_ids) == 0:
                ZillizCloudPipelineIndex.create_pipelines(
                    project_id = self.ZILLIZ_PROJECT_ID,
                    cluster_id=self.ZILLIZ_CLUSTER_ID,
                    api_key=self.ZILLIZ_TOKEN,
                    collection_name=config.pipeline.collection_name,
                    data_type = "doc",
                    language='CHINESE',
                    reranker= 'zilliz/bge-reranker-base',
                    embedding='zilliz/bge-base-zh-v1.5',
                    chunkSize=self.config.pipeline.chunk_size,
                    metadata_schema={"digest_from":"VarChar"}
                )
                pipeline_ids = self._list_pipeline_ids()
            self.index = ZillizCloudPipelineIndex(pipeline_ids=pipeline_ids, api_key=self.ZILLIZ_TOKEN)
        except Exception as e:
            print('(rag) zilliz pipeline 连接异常', str(e))
            exit()
        try:
            self._milvus_client = MilvusClient(
                uri=self.ZILLIZ_CLUSTER_ENDPOINT, 
                token=self.ZILLIZ_TOKEN 
            )
        except Exception as e:
            print('(rag) zilliz cloud 连接异常', str(e))

    def build_index(self, path, overwrite):
        config = self.config
        if not is_valid_url(path) or 'github' not in path:
            print('(rag) 不是一个合法的url，请尝试`https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/baihuasanguozhi.txt`')
            return
        if overwrite == True:
            self._milvus_client.drop_collection(config.pipeline.collection_name)
            pipeline_ids = self._list_pipeline_ids()
            self._delete_pipeline_ids(pipeline_ids)

            self._initialize_pipeline(self.service_context)

        if is_github_folder_url(path):
            urls = get_github_repo_contents(path)
            for url in urls:
                print(f'(rag) 正在构建索引 {url}')
                self.build_index(url, False)  # already deleted original collection
        elif path.endswith('.txt'):
            self.index._insert_doc_url(
                url=path,
                metadata={"digest_from": HistorySentenceWindowNodeParser.book_name(os.path.basename(path))},
            )
        else:
            print('(rag) 只有github上以txt结尾或文件夹可以被支持。')

    def build_query_engine(self):
        config = self.config
        self.query_engine = self.index.as_query_engine(
          search_top_k=config.pipeline.retrieve_topk)
        message_templates = [
            ChatMessage(content=QA_SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=QA_PROMPT_TMPL_STR,
                role=MessageRole.USER,
            ),
        ]
        chat_template = ChatPromptTemplate(message_templates=message_templates)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": chat_template}
        )
        self.query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = REFINE_PROMPT_TMPL_STR


    def delete_file(self, path):
        config = self.config
        if self._milvus_client is None:
            self._get_index()
        num_entities_prev = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        res = self._milvus_client.delete(collection_name=config.milvus.collection_name, filter=f"doc_name=='{path}'")
        num_entities = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        print(f'(rag) 现有{num_entities}条，删除{num_entities_prev - num_entities}条数据')

    def query(self, question):
        if self.index is None:
            self.get_index()
        if question.endswith("?") or question.endswith("？"):
            question = question[:-1]
        if self._debug is True:
            contexts = self.query_engine.retrieve(QueryBundle(question))
            for i, context in enumerate(contexts): 
                print(f'{question}', i)
                content = context.node.get_content(metadata_mode=MetadataMode.LLM)
                print(content)
            print('-------------------------------------------------------参考资料---------------------------------------------------------')
        response = self.query_engine.query(question)
        return response

    def _list_pipeline_ids(self):
        url = f"https://controller.api.gcp-us-west1.zillizcloud.com/v1/pipelines?projectId={self.ZILLIZ_PROJECT_ID}"
        headers = {
            "Authorization": f"Bearer {self.ZILLIZ_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        collection_name = self.config.pipeline.collection_name
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        pipeline_ids = {}
        for pipeline in response_dict['data']: 
            if collection_name in  pipeline['name']:
                pipeline_ids[pipeline['type']] = pipeline['pipelineId']
            
        return pipeline_ids

    def _delete_pipeline_ids(self, pipeline_ids):
        for pipeline_id in pipeline_ids:
            url = f"https://controller.api.gcp-us-west1.zillizcloud.com/v1/pipelines/{pipeline_id}/"
            headers = {
                "Authorization": f"Bearer {self.ZILLIZ_TOKEN}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            response = requests.delete(url, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(response.text)

