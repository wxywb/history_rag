# 史料 RAG 项目学习计划与实战作业（完善版）

基于对 `history_rag` 项目源码（`cli.py`、`executor.py`、`custom/`、`cfgs/`）的实际实现分析，这份计划在你原有结构上补充了“可量化验收”和“工程修复”维度，目标是让你不止会用，还能改、能排错、能扩展。

计划遵循**“先跑通骨架 -> 吃透 RAG 主链路 -> 掌握模型与参数 -> 完成工程化改造”**的顺序，每个阶段都包含源码阅读、探索任务、实战作业和验收标准。

---

## 0. 学习总目标（先对齐）

完成本计划后，你应具备以下能力：

1. 用 10 分钟讲清 `milvus` 与 `pipeline` 两种模式的端到端流程。
2. 独立完成一次“`build -> ask -> ask -d -> remove`”闭环。
3. 能解释并调优 `retrieve_topk`、`rerank_topk`、`window_size`、`temperature` 的影响。
4. 能新增一个 LLM 适配器并通过配置切换运行。
5. 能定位并修复至少 2 个项目内实际问题，并给出验证结果。

---

## 阶段一：破冰与解构 —— 走进 Python 工程与架构 (预计耗时: 1-2天)

**🎯 学习目标:**
不急于看懂所有的 AI 代码，先理清程序的“骨架”。掌握 Python 程序的入口模式、配置文件的读取、类的基础封装以及简单的交互逻辑。

**📖 核心源码阅读:**
- `cli.py` (命令行交互入口)
- `cfgs/config.yaml` (全局配置文件)
- `start_info.txt`（CLI 启动信息）

**🔍 探索任务:**
1. **程序的入口与启动**：打开 `cli.py`，拉到最底部的 `if __name__ == '__main__':`，看 `argparse` 是如何解析命令行参数的。
2. **配置如何流转**：看 `cli.py` 中的 `read_yaml_config()` 方法，了解 `yaml.safe_load()` 的运作，并弄懂 `EasyDict` 是什么（为什么要用它包裹字典）。
3. **模式分发与环境校验**：阅读 `CommandLine.run()` 和 `check_required_env()`，弄清模式分发前做了哪些环境变量检查（`OPENAI_API_KEY`、`ZILLIZ_*` 等）。
4. **命令解析边界**：阅读 `parse_input()`，记录每个命令允许的参数形态（如 `ask -d`、`build -overwrite`）。

**📝 实战测验 (作业):**
1. **修改启动问候语**：找到 `cli.py` 加载的 `start_info.txt` 并修改里面的 ASCII Art 或欢迎词，重新运行程序看它是否生效。
2. **新增一个占位模式**：在 `cli.py` 的 `run()` 方法中（`[milvus|pipeline]` 选项处），模仿现有逻辑添加一个名叫 `local_test` 的模式。当用户输入 `local_test` 时，打印一句 `"本地测验模式启动"` 然后通过 `break` 退出循环。
3. *(思考题)* 找找看 `cfgs/config.yaml` 里面存放了哪些信息？如果没有这个 yaml 文件，项目中要怎么写代码？使用配置文件有什么好处？

**✅ 阶段验收标准:**
1. 你可以口述 CLI 从启动到执行 `ask` 的完整调用链。
2. 你可以列出至少 4 个启动/运行依赖的环境变量与作用。
3. 你可以说明 `--cfg` 传参如何影响模型与检索配置。

---

## 阶段二：庖丁解牛 —— RAG 核心组件逐一击破 (预计耗时: 3-4天)

**🎯 学习目标:**
这是 RAG 的核心。你需要理解一个 txt 文本文档是如何变成向量存入数据库的，以及检索时数据库是如何工作的。学习 `llama_index` 的基础 API。

**📖 核心源码阅读:**
- `executor.py` 中的 `MilvusExecutor.build_index()` 函数和 `MilvusExecutor._get_index()` 函数。
- `custom/history_sentence_window.py` (文档切片的核心逻辑)。

**🔍 探索任务:**
1. **文档加载 (Document Loading)**：看 `build_index` 中 `FlatReader().load_data()` 是如何将长文本读取进内存的。
2. **文本切分 (Chunking & Node Parsing)**：深入看 `MilvusExecutor.__init__` 中实例化的 `HistorySentenceWindowNodeParser`。
   - **重点**：打开 `custom/history_sentence_window.py`，看看 `analyze_titles()` 怎么识别出“纪”和“传”（针对二十四史特色），以及它是如何给 Node（节点）打上 metadata（出处）标签的。
3. **Embedding 与存储 (Vector DB)**：梳理 `build_index` 方法，看 `nodes`（切分好的文本块）是如何交给 `VectorStoreIndex` 生成向量并存入 Milvus 的。
4. **检索与重排链路**：补读 `build_query_engine()`，看 `retrieve_topk` 与 `rerank_topk` 在流程里的先后关系。

**📝 实战测验 (作业):**
1. **Debug 切分效果**：通过打断点或者加 `print` 语句，在 `history_sentence_window.py` 的 `build_window_nodes_from_documents()` 的循环里，打印出被切分出的前 3 个 `node.text` 内容，看看原本的文本长什么样。
2. **修改窗口大小**：RAG 常用的策略是 "Sentence Window Retrieval"。在 `cfgs/config.yaml` 里，将 `window_size: 4` 改为 `window_size: 1`。重新执行 `build` 导入一个短文本，使用 `ask -d` (进入 Debug 模式)，观察终端输出的检索内容（context）有什么变化？（理论上返回的文本块会变短）。
3. **改变切分符号**：在 `executor.py` 中 `HistorySentenceWindowNodeParser.from_defaults()` 的 `sentence_splitter` 位置，把 `re.findall("[^,.;。？！]+[,.;。？！]?", text)` 改为仅按 `。？！` 切分，观察答案引用的变化。

**✅ 阶段验收标准:**
1. 你能解释 `window_size`、`retrieve_topk`、`rerank_topk` 的职责差异。
2. 你能用 `ask -d` 证明“召回结果变化”确实来自参数调整而不是模型随机性。
3. 你能给出一份简短实验记录（至少 3 组参数组合）。

---

## 阶段三：画龙点睛 —— 大模型交互与 Prompt 工程 (预计耗时: 2-3天)

**🎯 学习目标:**
理解 RAG 的“生成”部分。掌握如何将检索到的资料（Context）和用户的问题（Query）拼装成提示词（Prompt）喂给大语言模型（LLM）。

**📖 核心源码阅读:**
- `executor.py` 中的 `MilvusExecutor.build_query_engine()`。
- `executor.py` 顶部的全局 Prompt 变量 (`QA_PROMPT_TMPL_STR`, `QA_SYSTEM_PROMPT`)。
- `custom/llms/` 目录下的某一个 LLM 封装（如 `proxy_model.py` 或 `QwenLLM.py`）。

**🔍 探索任务:**
1. **寻找灵魂 Prompt**：细读 `QA_SYSTEM_PROMPT` 和 `QA_PROMPT_TMPL_STR`。看看作者在这个 Prompt 里施加了什么“紧箍咒”（比如要求模型“如果发现资料无法得到答案，就回答不知道”，以及要求写出“《书名》”）。
2. **Query Engine 的组装**：看 `build_query_engine()` 方法中，`ChatPromptTemplate` 是如何包装上述两个 Prompt 并更新进 `query_engine` 的。
3. **Refine 机制**：补读 `REFINE_PROMPT_TMPL_STR` 与 `_refine_template` 更新逻辑，理解“只有原答案是不知道才修正”的策略。
4. *(选做)* 看一看 `custom/llms/` 中的代码，体会一下面向对象编程中“多态”的魅力：无论你用哪家厂商的模型（OpenAI、Gemini、通义千问），对外暴露的接口都是一样的。

**📝 实战测验 (作业):**
1. **Prompt 调教**：修改 `QA_PROMPT_TMPL_STR`。加上一句指令：“请你用文言文甚至是古人的口吻来回答，显得你非常有学问。” 启动项目，问一个历史问题，看看 LLM 的回答风格是否发生了剧变。
2. **修改模型采样温度**：在 `config.yaml` 中，将 `temperature: 0.01` 改为 `temperature: 0.9`。这代表大模型的随机性/发散性变高。测试同一个问题问两次，看看答案的差异大不大。（RAG 场景通常要求 temperature 非常低以保证严谨，你可以亲自验证这种差异）。
3. **重排机制 (Rerank)**：查阅代码中 `SentenceTransformerRerank` 的使用（在 `executor.py` 顶部和初始化时）。想一想：为什么我们从向量数据库里 Retrieve 出 Top-K 文本后，还需要再进行一次 Rerank（重排）？它的意义是什么？

**✅ 阶段验收标准:**
1. 你能清楚解释三个 Prompt 的分工（System/QA/Refine）。
2. 你能给出一次“高温度导致稳定性下降”的对比证据。
3. 你能说明该项目如何在回答中加入证据出处。

---

## 阶段四：进阶与重构 —— 高阶优化与学以致用 (预计耗时: 3-5天)

**🎯 学习目标:**
脱离框架的束缚，开始自己动手修改甚至添加新功能。如果你能完成这部分，说明你已经完全吃透了这个项目。

**📖 核心源码阅读:**
- `gradioui.py`（如果有兴趣做 Web 界面）
- 整个 `executor.py` 逻辑。
- `custom/llms/*.py`（模型扩展点）
- `cfgs/config_*.yaml`（模型配置切换）

**🔍 探索与实战 (选做挑战作业):**

1. **[Web 界面实战] 点亮 UI**：
   研究 `gradioui.py` 文件，用命令 `python gradioui.py` 跑起 Web 服务。尝试在 UI 上加一个新的组件，比如一个 `slider`（滑块）小工具，用来在界面上直接调节 config 中的 `milvus.retrieve_topk`（取回多少条历史片段）的数值。
   
2. **[数据解析实战] 增加对 Markdown 的支持**：
   目前项目只支持 `.txt` 文件的读取（见 `build_index` 里的判断）。 
   试着利用 LlamaIndex 生态（如 `MarkdownReader` 或者手写处理逻辑），让 `build` 命令可以支持导入 `.md` 后缀的文档，并正确提取知识。

3. **[流程监控实战] 耗时统计器**：
   在 `executor.py` 的 `query()` 方法中找个合适的位置，使用 Python 的 `time` 模块，分别统计“检索相似片段花费了多少秒”和“大模型生成答案花费了多少秒”，并使用 `print()` 打印在控制台上。这对评估 AI 系统的性能非常有帮助。

4. **[模型扩展实战] 新增一个 LLM 配置**：
   参考 `custom/llms/proxy_model.py` 或 `QwenLLM.py`，新增一个最小可用适配器（如 `MyLLM.py`），并新增 `cfgs/config_myllm.yaml`。在 `cli.py --cfg` 下成功完成一次 `ask`。

5. **[工程修复实战] 修真实问题**：
   从以下问题中至少修 2 项，并记录复现和修复结果：
   - `cfgs/config_proxy_model.yaml` 出现明文 API Key（安全风险）。
   - `PipelineExecutor._delete_pipeline_ids()` 遍历了字典键而非 pipeline id 值。
   - `PipelineExecutor.build_index()` 在 `overwrite=True` 分支调用 `_initialize_pipeline(self.service_context)`，参数不匹配。
   - `PipelineExecutor.delete_file()` 使用 `config.milvus.collection_name`，需核对 pipeline 场景字段。

**✅ 阶段验收标准:**
1. 你提交了至少 1 次带代码修改的 commit（包含变更说明）。
2. 你完成了至少 2 项真实问题修复并给出验证步骤。
3. 你能在不改主流程的前提下切换到一个新的 LLM 配置运行。

---

## 阶段五：评估与复盘 —— 从“能跑”到“可信” (预计耗时: 1-2天)

**🎯 学习目标:**
建立可复用的评估方法，避免只凭主观感觉判断 RAG 效果。

**📖 核心实践内容:**
1. 制作一个最小评估集（建议 20 题）：
   - 事实题（可直接从史料找到）
   - 歧义题（需多段证据综合）
   - 无答案题（应输出“不知道”）
2. 设定指标：
   - 回答正确性
   - 引用准确性（出处是否匹配）
   - 稳定性（同问多次波动）
   - 响应耗时
3. 对比至少两组配置（如 `window_size=1 vs 4`、`retrieve_topk=20 vs 100`）。

**📝 实战测验 (作业):**
1. 形成一页评估记录表（参数、问题、答案、结论）。
2. 给出你的“默认推荐参数组合”与理由。
3. 列出仍未解决的风险点（如数据覆盖不足、模型幻觉残留）。

**✅ 阶段验收标准:**
1. 你能拿评估数据而不是感觉来解释参数选择。
2. 你能指出至少 2 个系统瓶颈及后续优化方向。

---

## 💡 学习小贴士

- **不要怕报错**。跑不通是 AI 时代程序员的常态！遇到 `import` 错误，检查 `virtualenv` 和 `requirements.txt`；遇到 LLM API 错误，检查网络和 Key。
- 善用 **Debug 模式** （即程序中的 `ask -d` 命令）。这个模式能帮你看到模型“背地里看到了什么材料”，这才是研究 RAG 秘密的关键。
- 每次改参数只改 1-2 个变量，并记录当次实验条件，否则很难定位因果关系。
- 如果你愿意，我可以在你做每一阶段时，按“你当前卡点”给你出下一步最小任务，不会让你陷入大而全阅读。
