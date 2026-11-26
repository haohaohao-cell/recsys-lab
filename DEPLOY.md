# 部署到 Streamlit Community Cloud

目标：拿到一个公开链接（形如 `https://xxx.streamlit.app`），任何人点开就能看，不用装任何东西。

## 已经准备好的东西

| 项 | 状态 |
|---|---|
| 精简数据 `assets/demo/`（10.1 MB） | ✅ 已随仓库提交，云端不用下 412MB 原始数据 |
| `requirements.txt` | ✅ 去掉了 torch / faiss / sentence-transformers，安装体积从 ~2GB 降到 ~200MB |
| 无 torch 时的降级 | ✅ SASRec 自动跳过，其余四个策略与全部图表照常 |
| 内存占用 | 约 400–500 MB（免费额度 1 GB 够用） |
| 启动时间 | 首次约 60–90 秒（构建 ItemCF / 语义模型），之后秒开 |

## 三步部署

**1. 打开** https://share.streamlit.io ，用 GitHub 账号 `haohaohao-cell` 登录并授权。

**2. 点 `Create app` → `Deploy a public app from GitHub`**，填：

| 字段 | 填什么 |
|---|---|
| Repository | `haohaohao-cell/recsys-lab` |
| Branch | `main` |
| Main file path | `app/dashboard.py` |

**3. 点 `Advanced settings`**，把 **Python version 选 3.12**（默认可能是 3.13，pandas/scipy 的轮子不一定齐）。

然后 `Deploy`，等 2–3 分钟。

## 部署后

拿到链接形如 `https://recsys-lab.streamlit.app`，把它贴到 README 顶部：

```markdown
**在线演示**：https://xxx.streamlit.app
```

## 已知限制（面试时主动说，比被问出来好）

- **云端没有 SASRec**：省内存没装 torch，所以策略切换只有 3 个。SASRec 的结果在「概览」和「诊断」页仍然完整显示——因为那些是从 `results/` 的 CSV 读的，不依赖模型。
- **免费实例会休眠**：15 分钟无人访问就睡，下次唤醒要 30 秒左右。演示前先自己点开一次。
- **数据是只读的精简包**：不能上传新数据、不能在线训练。

## 如果部署失败

| 现象 | 原因 | 处理 |
|---|---|---|
| 装依赖超时 | 网络 | 重试一次，或在 `Advanced settings` 里换区域 |
| `ModuleNotFoundError` | Python 版本不对 | 确认 3.12 |
| 启动后白屏 / 报错 | 缺 `assets/demo` | 确认仓库里 `assets/demo/*.parquet` 三个文件都在 |
| 内存超限 | 同时打开了 SASRec | 确认云端 `requirements.txt` 没有 torch |

## 想连 SASRec 一起上云

在 `requirements.txt` 里加 `torch==2.14.1`（CPU 版约 200MB 安装体积），并把
`results/sasrec.pt` 保留在仓库里。代价是内存 +300MB、冷启动 +40 秒——
免费额度下比较勉强，不建议。
