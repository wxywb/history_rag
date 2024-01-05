import logging
import llama_index
import sys
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index import set_global_service_context
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts import PromptTemplate
from llama_index.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.indices.query.schema import QueryBundle
from llama_index.callbacks import (
    CallbackManager,
    AimCallback,
    LlamaDebugHandler,
    CBEventType,
)
from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole
import llama_index
from  pymilvus import utility
from common import dataset, index_name, retrieve_debug, top_k, get_index_name, sanguo_augment, rerank_k
from bge_ranker import BGERerank  
from utils import SanguoAugmenter

def custom_rag_engine(vector_store):
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    postprocessor = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large", top_n=rerank_k
    )

    query_engine = index.as_query_engine(node_postprocessors=[
        postprocessor,
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ])

    query_engine._retriever.similarity_top_k=top_k
    qa_prompt_tmpl_str = (
        "请你仔细阅读相关内容，结合历史资料进行回答 (如果回答请引用原文,先给出回答，再贴上对应的原文，使用[]对原文进行标识),，如果发现资料无法得到答案，就回答不知道 \n"
        "搜索的相关历史资料如下所示.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "问题: {query_str}\n"
        "答案: "
    )
    message_templates = [
        ChatMessage(content="你是一个严谨的历史知识问答智能体，你会仔细阅读历史材料并给出准确的回答,你的回答都会非常准确，因为你在回答的之后，使用在[]内给出原文用来支撑你回答的证据.并且你会在开头说明原文是否有回答所需的知识", role=MessageRole.SYSTEM),
        ChatMessage(
            content=qa_prompt_tmpl_str,
            role=MessageRole.USER,
        ),
    ]
    chat_template = ChatPromptTemplate(message_templates=message_templates)
    refine_prompt_tmpl_str = ( 
           "你是一个历史知识回答修正机器人，你严格按以下方式工作"
            "1.只有原答案为不知道时才进行修正,\n"
            "2.修正的时候为了体现你的精准和客观，你非常喜欢使用[]将原文展示出来.\n"
            "3.如果感到疑惑的时候，就用原答案回答。"
            "新的知识: {context_msg}\n"
            "问题: {query_str}\n"
            "原答案: {existing_answer}\n"
            "新答案: "
     )
       
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": chat_template}
    )
    query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = refine_prompt_tmpl_str
    return query_engine


if __name__ == '__main__':
    dataset_name = 'sanguo'
    indices =  get_index_name(dataset_name)
    
    vector_store = MilvusVectorStore(
        host = "localhost",
        port = "19530",
        collection_name = indices[index_name]['collection'],
        dim=indices[index_name]['dim'])
    
    if 'bge' in index_name:
        embed_model = HuggingFaceEmbedding(model_name=indices[index_name]['model_name'])
        #service_context = ServiceContext.from_defaults(embed_model=embed_model)
        
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        llm = OpenAI(temperature=0.1, model="gpt-4")
        service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model, callback_manager=callback_manager)
        set_global_service_context(service_context)
    
    query_engine = custom_rag_engine(vector_store)
    query = '华雄是被谁杀死的?'
    
    if dataset_name == 'sanguo' and sanguo_augment is True:
        augmenter = SanguoAugmenter()
        augmenter.index_doc()

        response = augmenter.query(query)
        if response is not None:
            print('问题:', query)
            print('答案:', response)
            exit()
    
    if retrieve_debug is True:
        contexts = query_engine.retrieve(QueryBundle(query))
        for i, context in enumerate(contexts): 
            print(f'{query}', i)
            print(context.text)

    response = query_engine.query(query)
    print('问题:', query)
    print('答案:', response)

   


