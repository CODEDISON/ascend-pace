# 冻结镜像 vLLM + NPU 不兼容发现

## 测试

在冻结镜像（sha256:2e56022a…）中进行了两步诊断：

1. **NPU 硬件正常**：`torch.randn(100,100,device='npu:0')` + matmul + sync → PASS
2. **vLLM 引擎初始化失败**：`LLM(model=..., tensor_parallel_size=1, enforce_eager=True)` → `RuntimeError: Engine core initialization failed`（单卡、无 custom extension、无 multiproc）

## 结论

冻结镜像中的 vLLM 版本与 NPU 的引擎初始化路径不兼容。这**不是 lane 代码问题**，是环境限制。

**受影响的 lane**：
- #4 F 臂（PACE forced-SDMA via vLLM hook）— f_proof 8 轮修复均卡在 vLLM 层
- #11+#12 内存 campaign（三模型 WP-A/WP-B）— A0a 8 轮修复后仍卡在 WorkerProc

**不受影响的 lane**（不使用 vLLM，直接调 PACE/SHMEM）：
- #13 barrier ✅ 十组全 PASS
- #8 稀疏分布 ✅ 3/6 组 PASS
- #9 L 容量 ✅ 纯 CPU 分析

## 意义

1. 我们在 #4 F 臂和 #11+#12 上的 16 轮修复不是白费——它们排除了所有 lane 层面的问题，最终定位到镜像层
2. 如果要在设备上跑 vLLM 实验，需要更换容器镜像（使用 vLLM-Ascend 官方镜像或修复当前镜像的 vLLM 安装）
3. 当前冻结镜像的设计目的是确保 PACE/SHMEM 测量的可重复性，不是为了运行 vLLM 引擎
