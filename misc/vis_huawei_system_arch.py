# 导入所需的库
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Cpp, Python
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.gcp.compute import GCE
from diagrams.gcp.storage import GCS
from diagrams.aws.ml import SagemakerModel
# --- 已修正的导入路径 ---
from diagrams.onprem.client import User

# 设置图表属性
graph_attr = {
    "fontsize": "20",
    "bgcolor": "transparent",
    "splines": "spline",  # 使用曲线箭头
}

# 创建图表，文件名为 system_architecture.png
with Diagram("", show=False, filename="system_architecture", graph_attr=graph_attr, direction="TB"):
    # 1. 客户端
    client = User("用户 / 客户端浏览器")

    # 2. 定义三个主要层次
    with Cluster("应用层 (Application Layer)"):
        rag_app = Server("RAG问答系统\n(FastAPI + vLLM)")
        retrieval_app = Server("图像检索与对比系统\n(FastAPI + Jinja2)")
        apps = [rag_app, retrieval_app]

    with Cluster("核心引擎层 (Core Engine Layer)"):
        python_api = Python("Python 封装接口\n(DeltaNNGDatabase)")

        with Cluster("自研核心引擎 DeltaNNGClear (C++)"):
            cpp_searcher = Cpp("检索引擎模块\n(Searcher)")
            cpp_builder = Cpp("索引构建模块\n(Builder)")
            cpp_graph = Cpp("图结构与存储模块\n(Graph)")

            # 内部模块关系
            cpp_searcher >> Edge(style="dashed", color="grey") << cpp_graph
            cpp_builder >> Edge(style="dashed", color="grey") << cpp_graph

    with Cluster("基础资源层 (Infrastructure Layer)"):
        with Cluster("硬件平台"):
            kunpeng_cpu = GCE("鲲鹏 CPU")
            ascend_npu = GCE("昇腾 NPU")

        with Cluster("存储"):
            index_storage = PostgreSQL("索引文件存储\n(Disk/SSD)")
            dataset_storage = GCS("数据集 / fvecs文件")

        with Cluster("AI模型库"):
            llm_models = SagemakerModel("LLM 模型")
            embedding_models = SagemakerModel("嵌入模型")

    # 3. 定义各层之间的关系

    # 用户 -> 应用层
    client >> Edge(label="HTTP/HTTPS 请求") >> rag_app
    client >> Edge(label="HTTP/HTTPS 请求") >> retrieval_app

    # 应用层 -> 核心引擎层
    apps >> Edge(label="调用接口", color="darkgreen") >> python_api
    python_api >> Edge(label="调用C++核心", color="darkblue") >> cpp_builder
    python_api >> Edge(label="调用C++核心", color="darkblue") >> cpp_searcher

    # 应用层 -> 基础资源层 (模型调用)
    rag_app >> Edge(label="加载/推理", color="purple") >> ascend_npu
    rag_app >> Edge(label="加载", color="orange") >> llm_models
    apps >> Edge(label="加载/推理", color="orange") >> embedding_models

    # 核心引擎层 -> 基础资源层
    cpp_builder >> Edge(label="并行构建", color="darkred") >> kunpeng_cpu
    cpp_searcher >> Edge(label="并行检索 / SIMD", color="darkred") >> kunpeng_cpu
    cpp_builder >> Edge(label="读写索引", color="brown") >> index_storage
    cpp_searcher >> Edge(label="加载索引", color="brown") >> index_storage
    cpp_builder >> Edge(label="读取向量", color="brown") >> dataset_storage

print(
    "架构图代码已生成。请在本地环境中安装 'diagrams' 和 'graphviz' 后运行此代码，以生成 'system_architecture.png' 文件。")