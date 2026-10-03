# 研究工作台维护说明

2026-10-03 重构。使用 frontend-design skill 制定视觉方向，静态HTML/CSS/JS，无运行时第三方依赖或外部字体。

## 入口与数据

- `index.html`：统一工作台；hash导航为overview/experiments/results/methods/datasets。
- `assets/workbench.css`：冷白/钴蓝视觉系统，桌面侧栏、移动横向导航，响应式与键盘焦点。
- `assets/workbench.js`：数据加载、筛选、详情、条形图、表格与下载。
- `data/progress.json`：L20核验快照、历史结果与实验。
- `data/dual3090.json`：独立监控来源，不操作服务器。超过30分钟提示旧快照。
- `data/screening3090.json`：从公开双3090报告整理的60篇初筛汇总，不能与val300直接比较。
- `data/results/index.json`：原有8个历史数据集；JSON/CSV下载入口保留。

现有dual3090和两份demo页保留，首页整合其入口。刷新每5分钟读取公开JSON，不保证远端自动发布。浏览器读取时间绝不冒充采样时间。

## 安全与真实性

只公开聚合数据，不上传私人地址、SSH凭据、金标、逐条预测或适配器。数据通过HTML转义渲染；缺失值显示“未提供”。图表排除oracle探针和待核验32B，完整表格仍保留并注明性质。新训练模型没有评分前不列入成绩。

## 本地检查

从docs目录启动 `python -m http.server 8901 --bind 127.0.0.1`。

已使用Edge/Playwright检查：五个导航、搜索筛选、展开实验、数据集切换、390px移动端无页面级横向溢出；截图审查桌面与移动页面。JS另经过node --check。
