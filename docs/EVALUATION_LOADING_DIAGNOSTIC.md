# L20推理加载停滞排查（2026-10-03）

## 证据

- B1权重加载停在5/8分片（23:24），之后约18分钟日志与read_bytes不变。
- GPU约28.5GB、利用率0%；没有生成记录，无致命Python异常。
- py-spy Python栈停在vLLM `load_merged_column_weight → param_data.copy_`。
- native栈：`cuMemcpyHtoDAsync_v2(memory.c) → sleep → nanosleep`，定位到虚拟GPU拷贝hook内等待，而不是磁盘持续读取。
- 原引擎使用multiprocessing fork。官方vLLM文档说明fork与部分线程/CUDA依赖不兼容，支持`VLLM_WORKER_MULTIPROC_METHOD=spawn`。

## 处理与边界

- 仅终止本次评测进程树，原日志与启动manifest归档，没有预测输出被覆盖。
- 改为spawn重新加载；尚需观察权重加载和真实生成才能确认解决。
- 加入单阶段日志静默10分钟失败门槛，只停止自己创建的进程组，不无限等候。
- 未删除共享虚拟GPU缓存、未修改配额或禁用虚拟化hook。
- 安装py-spy仅用于只读线程栈诊断，没有升级torch/vLLM。

不能据此断言fork是最终根因。spawn如仍失败，应保留新栈并评估同进程HF推理兜底；如HF也在hook等待，应将诊断交给服务器管理员，不绕过调度暂停状态。
