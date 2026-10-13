# 变量、配置与秘密

## 当前状态

包内**不含任何密钥**。默认路径不需要任何凭据：

- **本地转写**（默认）：依赖可选的 `faster-whisper`；未安装时只是不能处理录音，其余功能不受影响。全程离线。
- **云端转写**（可选）：只在环境变量里**同时**出现 `TENCENTCLOUD_SECRET_ID` 和 `TENCENTCLOUD_SECRET_KEY` 时才启用。密钥由使用者自己申请和填写，只从环境变量读取，**不入库、不进仓库、不写日志**。

长期记忆默认使用操作系统用户数据目录；测试或高级用户可以覆盖目录。

## 环境变量

| 名称 | 使用者 | 作用域 | 来源 | 轮换 | 风险 |
| --- | --- | --- | --- | --- | --- |
| `MANMANTALK_MEMORY_DIR` | `scripts/memory_store.py` | 可选、本机 | 用户环境 | 不适用 | 指向私密本地目录；不得设为仓库或公共同步目录。旧名 `MENTALXTALK_MEMORY_DIR` 仍被识别（仅作兼容回退，新配置一律用新名） |
| `TENCENTCLOUD_SECRET_ID` | `scripts/transcribe.py` | 可选、本机 | 用户环境 | 建议定期 | 泄露可被冒用计费；**不得用主账号密钥**，应建子账号只授权语音识别 |
| `TENCENTCLOUD_SECRET_KEY` | `scripts/transcribe.py` | 可选、本机 | 用户环境 | 建议定期 | 同上；泄露即账号失守 |
| `TENCENTCLOUD_REGION` | `scripts/transcribe.py` | 可选、本机 | 用户环境 | 不适用 | 低；默认 `ap-guangzhou` |

两个密钥变量**必须成对存在**才启用云端。只设其中一个时脚本按「没有云端」处理，退回本地，不会拿半套凭据去连（已实测）。命令行可用 `--provider local` 强制本地、`--provider tencent` 强制云端，后者缺密钥直接报错。

## 脚本内置常量

### `scripts/transcribe.py`（转写）

**本地路径与通用：**

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `AVG_LOGPROB_FLOOR` | `-1.0` | 低于此值标记为「听不清」。作用于 `avg_logprob` **原值**，不是换算后的概率 |
| `HF_ENDPOINT` | `https://hf-mirror.com`（默认，可被同名环境变量覆盖） | 模型下载镜像 |
| `SUPPORTED` | `.mp3` `.wav` `.m4a` `.aac` `.flac` `.ogg` `.wma` `.mp4` `.amr` | 接受的音频格式 |

**腾讯云路径：**

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `TENCENT_HOST` | `asr.tencentcloudapi.com` | 接口域名 |
| `TENCENT_SERVICE` | `asr` | 签名用的服务名，必须与域名一致 |
| `TENCENT_VERSION` | `2019-06-14` | API 版本 |
| `TENCENT_CONTENT_TYPE` | `application/json` | **不带 charset**，必须与官方 SDK 逐字一致 |
| `TENCENT_HARD_LIMIT` | `5242880`（5 MB） | 单次直传的 base64 硬上限，来自接口文档 |
| `TENCENT_SAFE_LIMIT` | `4823449`（4.6 MB） | 脚本自留的安全线，低于硬限 |
| `LANG_TO_ENGINE` | `zh→16k_zh`、`en→16k_en`、`yue→16k_yue` | 语言到引擎的映射 |
| `BITRATE_LADDER` | opus `64/48/32/24/16`；mp3 `128/96/64/48/32` | 从高到低试编码的码率阶梯 |
| `BITRATE_FLOOR` | opus `24`；mp3 `32` | 低于此值宁可不降——**不靠压到听不清来换单段** |

`SourceType=1`（base64 直传）、`ChannelNum=1`、`SpeakerDiarization=1`、`SpeakerNumber=0`（自动）、`ResTextFormat=2` 写在 `tencent_create_task()` 里。

### `scripts/memory_store.py`（长期记忆）

以下常量在 `scripts/memory_store.py` 顶部定义，是数据边界的实际执行值：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `POLICY_VERSION` | `"1"` | 同意与数据策略版本 |
| `MAX_VALUE_CHARS` | `200` | 单个字段值的最大字符数 |
| `MAX_SOURCE_CHARS` | `200` | 来源说明的最大字符数 |
| `MAX_ROWS` | `200` | 全部作用域的条目总上限 |
| `MAX_OPERATIONS` | `20` | 可撤销操作历史条数 |

## 作用域与字段白名单

只有下列字段可以被写入。任何不在白名单内的字段会被拒绝，并返回可接受字段列表。

| 作用域 | 上限 | 允许字段 |
| --- | --- | --- |
| `guardian` | 20 | 关系类型、长辈年龄、长辈性别、教育理念、长辈自评 |
| `child` | 15 | 孩子年龄、孩子学段、孩子生理性别、孩子青春期状态 |
| `relationship` | 10 | 最近关系自评、主要冲突场景 |
| `scene` | 20 | 场景标签 |
| `level` | 10 | 历次等级 |

**硬性黑名单**：字段名里出现以下任一词，一律拒绝写入——原话、逐字稿、录音、姓名、名字、学校、班级、生日、出生、地址、电话、手机号、身份证、诊断。

## 来源类型与置信度

| 枚举 | 值 |
| --- | --- |
| `SOURCE_TYPES` | `user_explicit`、`user_report`、`record`、`tool`、`assistant_inference` |
| `CONFIDENCE_LEVELS` | `high`、`medium`、`low` |

## 命令接口

| 命令 | 作用 |
| --- | --- |
| `status` | 查看当前状态（是否启用、条数） |
| `enable --confirm` | 首次同意并启用 |
| `pause` / `resume` | 暂停 / 恢复 |
| `apply --json ...` / `apply --file ...` | 写入一条变更 |
| `undo --op-id ...` | 撤销指定操作 |
| `show --subject-id ...` | 查看某个对象的记忆 |
| `context --subject-id ... --max-chars ...` | 取出压缩上下文（500–8000 字符） |
| `forget <subject_id> --confirm` | 忘记某个对象 |
| `revoke --delete --confirm` | 撤回同意并删除 |
| `clear --confirm` | 清空全部 |

## 发布检查

- 确认仓库中没有录音、逐字稿、聊天导出、截图、家庭档案或任何可识别信息。
- 确认没有令牌、Cookie、设备码、私钥或 `.env` 文件；**确认没有腾讯云密钥**（`TENCENTCLOUD_*` 只应存在于使用者的环境变量里）。
- 确认包内不含任何过程文件（`draft/` 已于定稿后移出，归档在包外 `_archive/`）；`documentation/` 不属于运行时安装内容。
- 确认归档目录中若含真实案例，不得随包分发；归档仅限维护者回查。
- 确认参考资料只包含公开来源或有权分发的原创整理内容。
- 确认 `LICENSE` 中的非商业条款未被修改或移除。
- 确认 `scripts/transcribe.py` 不含硬编码路径、账号或密钥；转写产物（`.jsonl` / `.md` / `.raw.json`）与编码中间文件（`.chunk*.ogg` / `.probe.*`，用后即删）都落在使用者指定的输出目录，不进包。
- 确认转写临时文件在正常结束与失败退出两条路径上都会被清理。
- 若以后再加入别的外部服务，同样只从环境变量读凭据，不写入 Skill、README、工作流日志或示例。
