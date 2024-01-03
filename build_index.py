import logging
import llama_index
import sys
import argparse
from llama_index import ServiceContext, StorageContext
from llama_index import set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
import llama_index
from  pymilvus import utility
from common import dataset, index_name, get_index_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', choices=['sanguo', 'history24'], help='Specify the corpus to process (sanguo or history24).')
    args = parser.parse_args()

    dataset_name = args.corpus
    documents = SimpleDirectoryReader(dataset[dataset_name]['path']).load_data()

    indices =  get_index_name(dataset_name)
    
    vector_store = MilvusVectorStore(
        host = "localhost",
        port = "19530",
        collection_name = indices[index_name]['collection'],
        overwrite=True,
        dim=indices[index_name]['dim'])
    
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if 'bge' in index_name:
        embed_model = HuggingFaceEmbedding(model_name=indices[index_name]['model_name'])
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        set_global_service_context(service_context)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    query_engine = index.as_query_engine()
    response = query_engine.query("测试")

