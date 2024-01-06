dataset = {}
dataset['sanguo'] = {'path': "./data/sanguo/", 'name':'sanguo' }
dataset['history24'] = {'path': "./data/history_24/", 'name':'history24' }

#index_name = 'bge_small'
#index_name = 'openai'
index_name = 'bge_base'

def get_index_name(dataset_name):
    indices = {}
    indices['openai'] = {'dim':1536, 'collection': f'openai_{dataset[dataset_name]["name"]}'}
    indices['bge_small'] = {'dim':512, 'collection': f'bge_small_{dataset[dataset_name]["name"]}', 'model_name':"BAAI/bge-small-zh-v1.5"}
    indices['bge_base'] = {'dim': 768, 'collection': f'bge_base_{dataset[dataset_name]["name"]}', 'model_name':"BAAI/bge-base-zh-v1.5"}
    indices['bge_large'] = {'dim': 1024, 'collection': f'bge_base_{dataset[dataset_name]["name"]}', 'model_name':"BAAI/bge-large-zh-v1.5"}
    return indices
 
retrieve_debug = True

top_k = 200
rerank_k = 15
