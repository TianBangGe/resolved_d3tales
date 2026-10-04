# D3TaLES × ReSolved：架构迁移 baseline

本文件记录已经完成并验证的工作和实际结果。更新日期：2026-10-04。

## 1. 已完成项目与数据核对

- `ReSolved/` 保存作者的基础模型、参考数据和权重，作者的原始源码、数据和权重保持未修改。
- `data_redox/` 保存本项目的 D3TaLES 原始数据、处理后数据和固定划分。
- 本任务采用 `canonical_smiles` 作为分子结构输入、`reduction_potential` 作为唯一目标；仅使用 D3TaLES 数据重新训练。

| 数据 | 核对结果 |
| --- | --- |
| 原始 D3TaLES 表 | 35,777 条记录 |
| 中性分子处理表 | 30,102 条；canonical SMILES 唯一；目标无缺失；基态电荷均为 0 |
| 去重记录 | 30,090 条唯一记录；12 条记录标记为重复目标取平均 |
| random_seed42 / scaffold_seed42 | 每套 Train 24,081、Validation 3,010、Test 3,011；ID 完整覆盖 |
| scaffold_key 跨集合重叠 | scaffold 划分的 Train/Validation/Test 两两重叠均为 0 |

## 2. 已完成最小改动的模型适配

新增 `ReSolved/model/d3tales_model.py`，通过继承作者的 `MPNNModel`，复用原子的类型嵌入、键的类型嵌入、7 层 MPNN、残差更新和完整编码器前向流程。

活动模型的数据流为：

```text
canonical_smiles
  → 作者原有分子图编码（显式 H、原子/键特征、双向边）
  → MPNN × 7 + residual connections
  → 节点与边表示合并
  → 单个 Set Transformer
  → Linear head
  → reduction_potential
```

替换后的读出模块不包含 EA 或溶剂预测参数，不接收介电常数、折射率或五溶剂标签。保留作者原有的键特征切片规则。父类构造时曾初始化原读出模块，随后替换；该初始化消耗随机数，但原读出参数不保留在最终模型中。

新增 `ReSolved/d3tales_data.py`，沿用作者 `features.py`；目标保持原符号和原始数值尺度。非法开发分子直接报错，不静默丢弃固定划分中的 ID。

## 3. 已完成固定划分与 Test 封存接口

- 数据与划分通过 `dataset_id` 对齐，检查重复 ID 和完整覆盖；开发样本同时核对 canonical SMILES。
- Test 行在目标数值转换、分子图构建之前被跳过。
- 入口只创建 Train 和 Validation loader；没有 Test 评估参数。
- 使用 Validation MAE 选择最佳权重，记录 Validation MAE、RMSE、R²。
- 预测文件保留 `dataset_id`；`residual` 定义为真实目标减预测值。

## 4. 已完成单目标训练入口

新增 `ReSolved/run_d3tales.py`，使用单目标 L1/MAE loss 和 AdamW。训练模式的默认参数沿用作者设置：7 层、128 维、3 个 Set Transformer seed points、1 个 head、batch size 32、lr=1e-4、weight decay=1e-5、250 epochs。训练随机种子为 42，划分始终使用已存在的 seed42 文件。

入口已实现两种模式：默认 `smoke` 使用数据表中最先遇到的 128 个 Train 和 64 个 Validation 分子，只运行 1 轮、不持久化图缓存；显式 `--mode train` 使用完整开发集。小样本选择只用于验证执行流程，不是用于比较模型性能的划分。

输出已验证包括配置与输入/代码哈希、软件版本、训练历史、最佳权重、Validation 预测、指标及损失曲线。训练模式的缓存指纹包含数据、划分、图编码代码以及 RDKit/PyTorch/PyG 版本。

## 5. 已完成自动测试与独立代码审查

7 项测试通过，覆盖：

1. 与作者原版图编码一致、显式氢一致、目标符号正确。
2. 单目标输出形状、批处理一致性，以及所有活动参数的有限梯度。
3. 按 ID 对齐划分；即使 Test 标签和 SMILES 无效，开发加载仍正常完成。
4. 拒绝重复的划分 ID。
5. 冒烟样本上限生效，超出上限的开发分子不解析、不构图。
6. 非法开发 SMILES 报错且不静默丢弃。
7. 短训练可生成最佳权重、Validation 预测和损失曲线，不生成 Test 结果。

独立审查发现缓存版本指纹缺失，已经补入 RDKit/PyTorch/PyG 版本。作者代码存在的文档字符串转义告警不影响本次测试，原文件未为此修改。

## 6. 已完成本地运行验证

运行环境：Python 3.10.21、PyTorch 2.7.1+cu118、PyG 2.8.0.post1、RDKit 2022.09.5；GPU 为 RTX 3050 Laptop（4 GB）。

在确定仅做小样本本地冒烟的约定之前，两套划分各完成过一次全开发集的单轮执行验证；均成功构建 24,081 个 Train 和 3,010 个 Validation 分子图，没有静默删样本，没有构建 Test 图。

| 历史单轮运行 | Validation MAE | Validation RMSE | Validation R² | Train-mean 常数预测 MAE |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.659544 | 0.889160 | 0.260617 | 0.820049 |
| Scaffold | 0.636306 | 0.869870 | 0.211068 | 0.784866 |

这些结果是历史执行验证，不是收敛后的正式基线结果。不同验证集的分布不同，不能按两列误差直接判定哪个划分更容易泛化。指标使用数据表原始数值尺度；物理单位、参比电极和溶剂定义未在本次本地文件核对中得到确认。

更改运行默认值后，两套小样本 GPU 冒烟均已完成：Train 128、Validation 64、1 epoch，约 2 秒的训练与验证时间，损失与输出有限，权重、预测和曲线生成成功，Test 保持封存。小样本仅包含 4 个训练 batch，指标不用于评价模型能力。

实际验证过的本地命令（从 `ReSolved/` 执行）：

```powershell
& 'D:\Anaconda\envs\resolved\python.exe' -m unittest discover -v
& 'D:\Anaconda\envs\resolved\python.exe' -u run_d3tales.py --split random --output results_d3tales/smoke_random
& 'D:\Anaconda\envs\resolved\python.exe' -u run_d3tales.py --split scaffold --output results_d3tales/smoke_scaffold
```

## 7. 已完成运行约定与目录清理

项目约定已写入根目录 `AGENTS.md`：本地仅做测试和小规模冒烟；正式训练由用户在 Linux / RTX 4090 服务器执行；每完成并验证一步就更新本 README；清理过期或重复产物。

先前启动的本地正式训练及调度进程已停止；未完成的正式实验目录、后台训练脚本、图缓存和重复说明已删除。历史单轮验证仅保留配置、训练历史和指标；冒烟权重验证后已清理。原作者源码、数据、参考权重与 D3TaLES 正式数据保留。

Python 编译缓存已清理，包括作者仓库携带的 4 个 Python 3.11 `.pyc` 文件；这些是可重新生成的产物，不是模型源码。

| 当前文件/目录 | 用途 |
| --- | --- |
| `README.md` | 已完成工作及结果的统一记录 |
| `AGENTS.md` | 项目协作约定 |
| `ReSolved/README.md` | 作者原始模型说明 |
| `ReSolved/model/d3tales_model.py` | 单目标模型适配 |
| `ReSolved/d3tales_data.py` | D3TaLES 图构建与开发集加载 |
| `ReSolved/run_d3tales.py` | 默认冒烟、显式完整训练的入口 |
| `ReSolved/test_d3tales.py` | 9 项行为测试 |
| `ReSolved/run_baselines.py` | 使用当前 Python 环境顺序运行两套完整训练 |
| `ReSolved/results_d3tales/` | 保留下来的本地验证记录 |
| `data_redox/` | 正式任务数据与固定划分 |

## 8. 已完成 Linux 服务器冒烟验证

依据用户提供的服务器终端日志记录；未通过远程连接核查服务器文件或硬件型号。运行环境名为 `25wp_resolved`，工作目录为 `~/wp/Resolved/myJCIM/Resolved_D3/ReSolved`，入口指定 `--device cuda`。

实际完成的命令：

```bash
python -u run_d3tales.py --mode smoke --split random --device cuda --output results_d3tales/server_smoke_random
python -u run_d3tales.py --mode smoke --split scaffold --device cuda --output results_d3tales/server_smoke_scaffold
```

两套均为 Train 128、Validation 64、1 epoch，状态为 `complete`，`result_scope` 为 `smoke_subset`，`test_status` 为 `sealed`；日志明确表示没有构建 Test 分子图。

| 服务器冒烟 | Train MAE | Validation MAE | Validation RMSE | Validation R² | 单轮训练与验证时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 6.559695 | 6.298657 | 6.353117 | -54.132043 | 0.4 秒 |
| Scaffold | 6.437896 | 6.242113 | 6.303364 | -50.441373 | 0.3 秒 |

服务器结果与本地同配置小样本冒烟近似一致，Validation MAE 差异约为 2.3e-7（random）和 1.9e-6（scaffold）。这支持迁移后的执行流程一致；不代表模型已收敛。默认 batch size 32 下仅进行了 4 次参数更新，大误差与负 R² 表示当前预测差于验证集均值参照，不能据此判定正式 baseline 无效。

## 9. 已完成服务器 random 正式训练前 31 轮

记录日期：2026-10-04。依据用户提供的服务器终端日志，random 正式运行已完成 31 个 epoch，日志随后显示第 32 轮的 batch 600/753；这是进度快照，不是完整 250 轮的结束结果。

本次运行确认：`--mode train`、Train 24,081、Validation 3,010、CUDA、seed 42、batch size 32、lr=1e-4、weight decay=1e-5。每轮 753 个训练 batch，Test 保持封存且没有构图。

| 已完成轮次 | Train MAE | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: | ---: |
| 第 1 轮 | 0.800046 | 0.762563 | 0.921060 | 0.206612 |
| 第 28 轮：截至日志时 Validation MAE 最优 | 0.493271 | 0.503031 | 0.774895 | 0.438440 |
| 第 31 轮：日志中最后完整轮次 | 0.482496 | 0.516887 | 0.800606 | 0.400558 |

前 31 轮没有报错或非有限损失；训练 MAE 整体下降，Validation MAE 有波动但最优值持续改善。这说明当前模型在完整开发数据上已学习到有效信号，尚不能仅据该片段判断收敛或确定最终性能。最佳模型按 Validation MAE 选择；表中的最优 MAE、RMSE、R² 来自同一个第 28 轮，不混用不同轮次的指标。

## 10. 已完成独立 Git 仓库整理

项目根目录已建立独立 Git 仓库，默认分支为 `main`，覆盖根目录 README、协作约定、ReSolved 源码和 `data_redox/`。用户指定远程仓库为 [TianBangGe/resolved_d3tales](https://github.com/TianBangGe/resolved_d3tales)。

作者参考来源为 [grynova-ccc/ReSolved](https://github.com/grynova-ccc/ReSolved)，本地参考版本为 `1206f6aa014a28a8938c0c332d7c9052bef3e16f`。原作者 Git 历史已保存在本地根仓库的 `.git/upstream-reference/`，源码作为普通项目文件管理，避免嵌套仓库导致新增代码未上传。历史备份属于本地元数据，不随新仓库上传。

根目录 `.gitignore` 已统一排除训练结果、权重、图缓存、Python 编译缓存、本机配置和凭据文件。作者的 `ReSolvedData.csv`、分子生成 notebook 与 `evo_gen/` 仅在本地保留作为参考，不进入新仓库；D3TaLES 正式数据和固定划分纳入版本控制。重复的子目录忽略文件已删除。

`.gitattributes` 已配置源码使用 LF，并禁止 Git 自动转换 CSV 换行符，以保留正式数据的原始字节与哈希。已有划分文件的目录占位文件 `.gitkeep` 已清理。

2026-10-04 已完成首次提交并推送至新仓库 `main` 分支，初始提交为 `36b7c90`（`Initialize D3TaLES single-target ReSolved baseline`）。共纳入 26 个文件，包含项目说明、源码、D3TaLES 原始/处理数据及 random/scaffold 划分；训练结果、模型权重与缓存未上传。

## 11. 已完成两套完整训练的顺序运行脚本

2026-10-04，依据用户本次明确允许本地完整训练的要求，新增 `ReSolved/run_baselines.py`。脚本使用 `sys.executable` 调用当前已激活环境中的 Python，先运行 random，成功结束后再运行 scaffold；两套默认均为完整 Train/Validation、250 epochs、CUDA、seed 42、batch size 32、lr=1e-4、weight decay=1e-5，Test 继续封存。

两套输出分别位于 `ReSolved/results_d3tales/local_full_baseline/random_seed42_250epochs/` 和 `scaffold_seed42_250epochs/`。脚本在启动前检查两套输出目录，拒绝覆盖已有结果；任一训练失败则停止后续执行。支持 `--dry-run` 打印调用命令，不创建结果或启动训练。

新增的两项测试验证当前 Python 环境、两套完整训练参数和第二套目录冲突时的提前拒绝行为；全套 9 项测试通过。已实际执行 dry-run，确认生成的两条命令正确；本次脚本验证没有启动完整训练。

## 12. 已完成本机两套 250 轮 baseline 训练与结果核对

两套结果已从 `ReSolved/results_d3tales/local_full_baseline/` 读取并核对。random 和 scaffold 均完整记录 epoch 1–250，状态为 `complete`，每套 Train 24,081、Validation 3,010，使用相同模型、训练超参数与 seed 42；参数量为 2,461,919。配置记录 GPU 为 RTX 3050 Laptop。

| Validation 最优模型 | 最佳 epoch（按 MAE） | MAE | RMSE | R² | 相对 Train-median 常数预测的 MAE 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 85 | 0.480343 | 0.768081 | 0.448272 | 40.6% |
| Scaffold | 54 | 0.524186 | 0.792096 | 0.345837 | 31.9% |

各行指标均来自同一 MAE 最优 checkpoint。指标采用数据原始数值尺度，均为 Validation，Test 仍封存。这是单个训练种子的结果，不代表多次运行平均值或最终 Test 性能。

核对结果：两套预测各有 3,010 个分子 ID，与各自 Validation 清单完全一致；与各自 Test 清单的 ID 交集为 0。预测文件中的目标和 SMILES 与正式处理表一致（目标允许 float32 存储的舍入误差）。从预测逐行重算 MAE、RMSE、R² 与保存指标一致；配置中的数据、划分和模型/训练代码哈希与当前文件一致。

训练曲线显示过拟合：random 的第 250 轮 Train MAE=0.199722、Validation MAE=0.505312；scaffold 的第 250 轮 Train MAE=0.201033、Validation MAE=0.554938。后期训练误差继续下降，验证误差停滞或回升，最佳验证模型分别出现在第 85 和第 54 轮。当前保存的 `best_model.pth` 按最优 Validation MAE 选择，结果文件也对应该最佳模型。

逐分子残差分析结果：

| 最优模型 Validation 误差分布 | Random | Scaffold |
| --- | ---: | ---: |
| 绝对误差中位数 | 0.225998 | 0.264086 |
| 绝对误差 P90（nearest-rank） | 1.355111 | 1.420694 |
| 绝对误差 P95（nearest-rank） | 1.783529 | 1.731258 |
| 最大绝对误差 | 4.660250 | 4.253811 |
| 绝对误差大于 1 的分子数 | 491 / 3,010 | 586 / 3,010 |
| 最大误差约 5% 样本对平方误差总和的贡献 | 46.2% | 39.9% |

模型相对常数预测有明确改善，但解释的验证目标方差仍有限，且有长尾大误差样本。scaffold 的误差更高、R² 更低，与跨骨架泛化更困难相符；两套 Validation 的分布不同，不能将差异全部归因于划分方式。现有结果尚未检验描述符互补信息，不据此归因于某类描述符或标签问题。

训练历史记录的累计训练与验证时间为 random 3.52 小时、scaffold 3.43 小时，不含初始构图等开销。本次检查未重新训练、未评估 Test，保留了两套最佳权重、配置、完整历史、预测和曲线作为 baseline 记录。

## 13. 已完成 baseline 性能诊断

新增 `ReSolved/diagnose_baseline.py`，完成既有 Validation 预测的分组残差分析、开发数据标签核对，以及 32 个 random Train 分子的拟合探针。结果保存于 `ReSolved/results_d3tales/diagnostics/diagnostics.json` 和 `train_only_fit.csv`；未修改基础模型、正式训练结果或权重，未评估 Test。

小样本探针沿用原模型与 AdamW（lr=1e-4、weight decay=1e-5），固定 32 个训练分子执行 300 次更新。训练模式 MAE 从 6.860390 降至 0.053997，评估模式下同一训练子集 MAE 从 6.875039 降至 0.061384。诊断前向保存并恢复 BatchNorm 缓冲区，避免测量改变运行统计量；未保存 checkpoint。结果表明模型能够拟合该训练子集，不证明完整数据泛化良好。

特征流核查确认：作者 `mpnn_model.py` 使用键类型 embedding，但数值键特征切片重复包含键类型，并跳过 `is_conjugated`。在拟合探针中翻转全部键的共轭列，预测最大变化为 0，验证该列未进入有效计算。此行为来自原作者代码；其对任务性能的影响尚未通过对照训练验证，也不能认为图结构完全不包含共轭信息。

按目标原始数值尺度分组的 Validation MAE：

| 目标区间 | Random 样本数 / MAE | Scaffold 样本数 / MAE |
| --- | ---: | ---: |
| [6, 7) | 1,222 / 0.2833 | 1,246 / 0.3004 |
| [8, 9) | 324 / 0.9448 | 321 / 1.1300 |
| [9, 12) | 89 / 1.6429 | 56 / 2.1946 |

高目标区域明显低估，低目标区域则倾向高估，预测向中间收缩。目标 >=9 的分子仅约占各套 Train 的 3.2% / 3.4%；结合第 12 节后期过拟合，当前证据指向尾部样本与泛化问题。不能据此删除高误差样本或断定标签错误。

来源分组中，Zinc 的 Validation MAE 为 0.533 / 0.554，csd 为 0.242 / 0.302（random / scaffold）；来源与目标分布、分子结构可能混杂，不能直接归因于来源噪声。SMARTS 分组未显示腈类或酰亚胺整体劣于全体样本；部分结构组样本很少，分组存在重叠。

原始标签核对仅使用不属于任一 Test 的开发分子：检查的 24,413 条原始记录中，`reduction_potential` 与 `solv_reduction_potential` 数值相同。官方 [D3TaLES 计算源码](https://d3tales.github.io/d3tales_api/_modules/d3tales_api/Calculators/calculators.html) 的还原电位计算涉及能量、校正项、溶剂化贡献与参比电位，不能简单等同于 EA 加固定常数。本次未据重复字段推定导出数据的溶剂、参比或物理单位，也未更换预测目标。

## 14. 已完成键共轭特征对照入口与本地冒烟

在 D3TaLES 模型和训练入口增加 `--bond-features legacy|conjugation`，默认仍为 `legacy`，保留已有 baseline 的行为。`conjugation` 将数值键输入从 `[bond_type, in_ring, ring_size]` 替换为 `[is_conjugated, in_ring, ring_size]`，键类型 embedding 仍保留。原作者 `features.py`、`mpnn_model.py`、`mpnn_layer.py` 未修改，图编码、7 层消息传递、残差、Set Transformer、预测头及 2,461,919 个参数均保持原配置。

`run_d3tales.py` 的 `config.json` 已记录特征模式与数值键特征名称；`run_baselines.py` 支持将同一特征模式传递给 random 和 scaffold。两种模式的权重形状相同，但输入语义不同，解释 checkpoint 时必须结合对应配置，不能将旧权重切换模式后直接当作新模式训练结果。

全套 10 项测试通过。新增行为验证：旧模式翻转共轭列时预测完全一致；新模式预测会变化；新模式将共轭位置恢复为键类型数值后与旧模式预测完全一致；原始图未被修改，两模式参数量一致，新模式反向梯度有限。顺序运行脚本的 dry-run 已验证两套命令均包含 `--bond-features conjugation`，未启动完整训练。

实际完成两套 CUDA 冒烟，每套 Train 128、Validation 64、1 epoch：

| Conjugation 冒烟 | Validation MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Random | 6.320282 | 6.374953 | -54.511676 |
| Scaffold | 6.262499 | 6.323945 | -50.777851 |

输出位于 `ReSolved/results_d3tales/conjugation_smoke_random/` 和 `conjugation_smoke_scaffold/`，均完成且 Test 封存。这些指标只用于检查小样本执行流程，不代表收敛性能；本次没有完整训练或证明精度提升，原有 250 轮 baseline 结果保留。

## 15. 已补充中文注释与提交约定

项目协作约定已明确 Git 提交说明使用中文，新增或修改代码添加详细中文注释。本次已为特征模式、checkpoint 输入语义、残差计算、Test 排除规则、BatchNorm 诊断状态恢复及顺序训练的目录检查补充说明。补充注释后重新执行全套 10 项测试，全部通过；差异格式检查通过。
