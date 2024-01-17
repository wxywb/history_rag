from executor import MilvusExecutor
from executor import PipelineExecutor

import yaml
from easydict import EasyDict

def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)

class CommandLine():
    def __init__(self):
        self._mode = None
        self._executor = None

    def show_start_info(self):
        with open('./start_info.txt') as fw:
            print(fw.read())

    def run(self):
        self.show_start_info()
        while True:
            conf = read_yaml_config('config.yaml')
            print('(rag) 选择[milvus|pipeline]方案')
            mode = input('(rag) ')
            if mode == 'milvus':
                self._executor = MilvusExecutor(conf) 
                print('(rag) milvus模式已选择')
                print('  1.使用`build data/history_24/baihuasanguozhi.txt`来进行知识库构建。')
                print('  2.已有索引可以使用`ask`进行提问, `-d`参数以debug模式进入。')
                print('  3.删除已有索引可以使用`remove baihuasanguozhi.txt`。')
                self._mode = 'milvus'
                break
            elif mode == 'pipeline':
                self._executor = PipelineExecutor(conf)
                print('(rag) pipeline模式已选择, 使用`build https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/baihuasanguozhi.txt`来进行知识库构建。')
                print('  1.使用`build https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/baihuasanguozhi.txt`来进行知识库构建。')
                print('  2.已有索引可以使用`ask`进行提问, `-d`参数以debug模式进入。')
                print('  3.删除已有索引可以使用`remove baihuasanguozhi.txt`。')
                self._mode = 'pipeline'
                break
            elif mode == 'quit':
                self._exit()
                break
            else:
                print(f'(rag) {mode}不是已知方案，选择[milvus|pipeline]方案,或者quit退出。')
        assert self._mode != None
        while True:
            command_text = input("(rag) ")
            self.parse_input(command_text)

    def parse_input(self, text):
        commands = text.split(' ')
        if commands[0] == 'build':
            if len(commands) == 3:
                if commands[1] == '-overwrite':  
                    print(commands)
                    self.build_index(path=commands[2], overwrite=True)
                else:
                    print('(rag) build仅支持 `-overwrite`参数')
            elif len(commands) == 2:
                self.build_index(path=commands[1], overwrite=False)
        elif commands[0] == 'ask':
            if len(commands) == 2:
                if commands[1] == '-d':
                    self._executor.set_debug(True)
                else: 
                    print('(rag) ask仅支持 `-d`参数 ')
            else:
                self._executor.set_debug(False)
            self.question_answer()
        elif commands[0] == 'remove':
            if len(commands) != 2:
                print('(rag) remove只接受1个参数。')
            self._executor.delete_file(commands[1])
            
        elif 'quit' in commands[0]:
            self._exit()
        else: 
            print('(rag) 只有[build|ask|remove|quit]中的操作, 请重新尝试。')
            
    def query(self, question):
        ans = self._executor.query(question)
        print(ans)
        print('+---------------------------------------------------------------------------------------------------------------------+')
        print('\n')

    def build_index(self, path, overwrite):
        self._executor.build_index(path, overwrite)
        print('(rag) 索引构建完成')

    def remove(self, filename):
        self._executor.delete_file(filename)
        
    def question_answer(self):
        self._executor.build_query_engine()
        while True: 
            question = input("(rag) 问题: ")
            if question == 'quit':
                print('(rag) 退出问答')
                break
            elif question == "":
                continue
            else:
                pass
            self.query(question)

    def _exit(self):
        exit()

if __name__ == '__main__':
    cli = CommandLine()
    cli.run()

