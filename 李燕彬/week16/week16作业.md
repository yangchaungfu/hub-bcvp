## 20260208-week16-第十六周作业

### 作业内容和要求：安装neo4j实践知识图谱文档过程

## 一、Neo4j 安装步骤

### 1. 下载 Neo4j
- 访问 [Neo4j 官方网站](https://neo4j.com/download/)
- 选择 Community Edition（社区版，免费）
- 下载对应操作系统的版本（macOS 选择 macOS 版本）

### 2. 安装 Neo4j
- **macOS 安装**：
  - 打开下载的 .dmg 文件
  - 将 Neo4j 拖动到 Applications 文件夹
  - 或者使用 Homebrew 安装：`brew install neo4j`

### 3. 启动 Neo4j
- **图形界面启动**：
  - 打开 Applications 中的 Neo4j Desktop
  - 创建一个新的项目
  - 添加一个新的数据库
  - 设置密码为 `demo`（与代码中的配置一致）
  - 启动数据库

- **命令行启动**（如果使用 Homebrew 安装）：
  ```bash
  neo4j start
  ```

### 4. 验证安装
- 打开浏览器访问 `http://localhost:7474`
- 输入用户名 `neo4j` 和密码 `demo`
- 看到 Neo4j 浏览器界面说明安装成功

## 二、环境配置与依赖安装

### 1. 安装 Python 依赖
在项目目录下执行：
```bash
pip install py2neo
```

### 2. 检查项目文件
确保项目目录结构如下：
```
week16 知识图谱问答/
└── kgqa_base_on_sentence_match/
    ├── build_graph.py
    ├── triplets_head_rel_tail.txt
    ├── triplets_enti_attr_value.txt
    └── kg_schema.json
```

## 三、构建知识图谱

### 1. 准备数据
项目已经提供了两种三元组数据：
- `triplets_head_rel_tail.txt`：实体-关系-实体三元组
- `triplets_enti_attr_value.txt`：实体-属性-属性值三元组

### 2. 执行构建脚本
在 `kgqa_base_on_sentence_match` 目录下执行：
```bash
python build_graph.py
```

### 3. 构建过程说明
`build_graph.py` 脚本会：
1. 连接到本地 Neo4j 数据库
2. 读取三元组数据文件
3. 处理实体标签（如从括号中提取标签）
4. 生成 Cypher 查询语句
5. 执行查询构建知识图谱
6. 生成 `kg_schema.json` 文件记录图谱结构

## 四、验证知识图谱

### 1. 在 Neo4j 浏览器中验证
- 打开 `http://localhost:7474`
- 输入用户名 `neo4j` 和密码 `demo`
- 执行以下 Cypher 查询：
  ```cypher
  MATCH (n) RETURN n LIMIT 25
  ```
- 查看生成的节点和关系

### 2. 查看图谱统计信息
执行以下查询：
```cypher
// 统计节点数量
MATCH (n) RETURN count(n) AS node_count

// 统计关系数量
MATCH ()-[r]->() RETURN count(r) AS relationship_count

// 查看所有标签
MATCH (n) RETURN DISTINCT labels(n) AS labels

// 查看所有关系类型
MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationships
```

## 五、知识图谱应用

项目中的 `graph_qa_base_on_sentence_match.py` 文件是基于句子匹配的知识库问答实现。可以执行该脚本进行问答测试：

```bash
python graph_qa_base_on_sentence_match.py
```

## 六、常见问题与解决方案

### 1. 连接数据库失败
- 检查 Neo4j 是否正在运行
- 确认密码是否设置为 `demo`
- 检查连接地址是否正确（默认 `http://localhost:7474`）

### 2. 数据导入失败
- 检查数据文件格式是否正确
- 确保文件编码为 UTF-8
- 查看控制台错误信息

### 3. 内存不足
- Neo4j 默认配置可能需要调整
- 编辑 `neo4j.conf` 文件，修改内存配置

## 七、项目数据说明

本项目构建的是关于周杰伦的知识图谱，包含：
- **实体**：周杰伦、歌曲、专辑、电影、人物等
- **关系**：演唱、创作、导演、制作等
- **属性**：歌曲语言、风格、所属专辑等

