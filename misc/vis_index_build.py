from diagrams import Diagram, Cluster, Edge
from diagrams.gcp.storage import GCS
from diagrams.onprem.database import PostgreSQL as DB
from diagrams.gcp.analytics import Dataflow, BigQuery

# --- 沿用之前的字体和样式设置 ---
graph_attr = {
    "fontsize": "20",
    "bgcolor": "transparent",
    "splines": "spline",
}
node_attr = {
    "fontsize": "16",
}
edge_attr = {
    "fontsize": "12",
}

# 创建图表
with Diagram(
        "",
        show=False,
        filename="index_build_process",
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        direction="LR"
):
    # 1. 流程起点：原始数据
    raw_data = GCS("原始向量数据集")

    # 将整个构建流程包裹在一个大的Cluster中
    with Cluster("索引构建流程 (Builder Module)"):
        # 2. 初始化阶段
        with Cluster("1. 初始化阶段"):
            calc_entrypoint = Dataflow("计算入口点\n(质心)")
            random_init = Dataflow("随机分配初始邻居")
            init_steps = [calc_entrypoint >> random_init]

        # 初始化阶段的产物
        initial_graph = DB("初始图结构\n(连接不完善)")

        # 3. 核心迭代优化阶段
        with Cluster("2. 核心迭代优化 (循环执行)"):
            # 将“遍历所有节点”抽象为一个处理单元
            for_each_node = BigQuery("遍历图中所有节点")

            # 定义循环内部的核心操作
            beam_search = Dataflow("Beam Search\n寻找候选邻居")
            pruning = Dataflow("邻居选择与剪枝策略\n(核心)")

            # 定义循环内部的流程
            for_each_node >> beam_search >> pruning

        # 4. 流程终点：最终产物
        final_graph = DB("最终优化的导航图")

        # 5. 连接所有阶段
        raw_data >> init_steps
        init_steps >> initial_graph

        # 从初始图进入迭代循环
        initial_graph >> for_each_node

        # 关键：描绘迭代循环的过程
        # 筛选完成后，更新了图，进入对下一个节点或下一轮的迭代
        pruning >> Edge(
            label="下一节点 / 下一轮迭代",
            color="darkred",
            style="dashed"
        ) >> for_each_node

        # 描绘循环结束，产出最终结果的路径
        pruning >> Edge(
            label="构建完成 (收敛)",
            color="darkgreen",
            style="bold"
        ) >> final_graph

print("索引构建流程图代码已生成。请在本地环境中运行此代码，以生成 'index_build_process.png' 文件。")