# Concern #8 设备级发现：syn-diag-512 早期 Case 触发 AICPU 507018

## 发现

在 P16 同步稀疏分布测量中，`syn-diag-512`（对角主导 90% self-traffic、仅 ~1.6% peer 流量的合成 case）作为 SHMEM 初始化后**首个/早期 PACE 调用**时，触发 HCCL AICPU 内核 `RunAicpuRpcSrvLaunchV2_alltoallv` 异常（错误码 507018，"execute kernel param invalid"）。全部 16 rank 同时崩溃。

## 证据链

| 实验 | 结果 | 说明 |
|------|------|------|
| job0（offset 0，首 case qwen-038-dispatch，dense） | PASS × 3 bindings | 真实 dense case 作为首笔流量安全 |
| job1（offset 3，首 case qwen-021-combine，dense） | PASS × 3 bindings | 同上 |
| job2（offset 6，首 case olmoe-050-dispatch，dense） | PASS × 3 bindings | 同上 |
| **job3（offset 9，首 case syn-diag-512，90% self）** | **FAIL × 4 次** | 4 次可复现，全 rank exit=1，ERR99999/aicpu 507018 |
| barrier A1（同期设备健康检查） | PASS | 设备功能正常，排除硬件故障 |
| 同一 syn-diag-512 在 job0-2 的后续位置 | PASS | 非_case 本身有毒_，是位置/序列依赖 |
| priming 1 次（dense PACE call 后再跑 syn-diag-512） | syn-diag-512 仍 FAIL | 单次暖通不足 |
| priming 5 次（dense PACE call × 5 后再跑） | syn-diag-512 仍 FAIL | 5 次暖通仍不足 |
| job0 带 priming（synthetic uniform） | job0 本身 FAIL | 合成 priming 流量自身触发问题 |

## 推断的最优解释

这不是"极稀疏冷启动"或"暖通不足"问题（priming 假设被推翻），而是**极端对角主导（90% self）的合成 case 几何**与 SHMEM AICPU 内核初始化序列之间的根本交互。可能的机制：近自环 alltoallv 模式导致 AICPU 队列管理器收到无效的 channel 分配参数（对角流量不产生跨 die 的 SDMA 传输，队列管理器可能未正确初始化 channel 状态）。

同一 case 在 ~160 个前置 PACE 调用后安全运行（job0-2 的后续位置），表明需要一定规模的**跨 peer 真实流量**才能完成队列初始化——但这不是简单的"暖通次数"问题（5 次不够）。

## 科学意义

1. 这是 PACE/SHMEM 部署的真实边界条件：极端稀疏/对角模式在冷启动时可能触发设备级异常
2. 对于生产 MoE serving（通常前几步就有 dense 流量），此问题不太可能触发
3. 但对于诊断/测试场景（直接用合成稀疏矩阵测试），这是一个已知限制

## 影响

- job3（rotation offset 9）不可运行
- job4-5 被严格组序连带阻塞
- 3/6 组可运行（job0-2），方向判定基于 3 组
- syn-diag-512 在 job0-2 后续位置的数据正常（位置依赖，非 case 缺陷）

## 分类

设备级限制（非代码缺陷、非可修复 bug——是 HCCL AICPU 内核与特定 alltoallv 模式的交互）。
