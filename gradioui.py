from executor import MilvusExecutor
from executor import PipelineExecutor

import gradio as gr

from cli import CommandLine, read_yaml_config  # 导入 CommandLine 类

resolutions = ["milvus", "pipeline"]

build_tasks = ["构建索引", "删除索引"]
query_tasks = ["提问", "提问+返回检索内容"]


class GradioCommandLine(CommandLine):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config_path = cfg

    def index(self, task, path, overwrite):
        if task == "构建索引":
            self._executor.build_index(path, overwrite)
            return "索引构建完成"
        elif task == "删除索引":
            self._executor.delete_file(path)
            return "删除索引"

    def query(self, task, question):
        if task == "提问":
            return self._executor.query(question)
        elif task == "提问+返回检索内容":
            self._executor.set_debug(True)
            return self._executor.query(question)


def initialize_cli(cfg_path, resolution):
    global cli_instance
    cli_instance = GradioCommandLine(cfg_path)
    conf = read_yaml_config(cli_instance.config_path)
    if resolution == "milvus":
        cli_instance._executor = MilvusExecutor(conf)
        cli_instance._mode = "milvus"
    else:
        cli_instance._executor = PipelineExecutor(conf)
        cli_instance._mode = "pipeline"
    cli_instance._executor.build_query_engine()
    return "CLI 初始化完成"


with gr.Blocks() as demo:
    # 初始化
    gr.Interface(fn=initialize_cli,
                 inputs=[gr.Textbox(
                     lines=1, value="cfgs/config.yaml"),
                     gr.Dropdown(resolutions, label="索引类别", value="milvus")],
                 outputs="text",
                 submit_btn="初始化", clear_btn="清空")
    # 构建索引
    gr.Interface(fn=lambda command, argument, overwrite: cli_instance.index(command, argument, overwrite),
                 inputs=[gr.Dropdown(choices=build_tasks, label="选择命令", value="构建索引"),
                         gr.Textbox(label="路径"), gr.Checkbox(label="覆盖之前索引")], outputs="text",
                 submit_btn="提交", clear_btn="清空")

    # 提问
    gr.Interface(fn=lambda command, argument: cli_instance.query(command, argument),
                 inputs=[gr.Dropdown(choices=query_tasks, label="选择命令", value="提问"),
                         gr.Textbox(label="问题")], outputs="text",
                 submit_btn="提交", clear_btn="清空")
    with open("docs/web_ui.md", "r", encoding="utf-8") as f:
        article = f.read()
    gr.Markdown(article)

if __name__ == '__main__':
    # 启动 Gradio 界面
    demo.launch()
