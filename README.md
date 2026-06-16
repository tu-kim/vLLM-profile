# vLLM-profile

vLLM **0.18.0** 내부 동작 프로파일링 addon — **DeepSeek V3** (MLA + fine-grained MoE)와
**Llama 3.1 405B** (GQA + dense MLP)의 MoE / Attention 내부 동작을 **vLLM 소스 수정 없이**
monkey-patch로 계측합니다.

대상 환경: H100 × 8 × 2대 = 16 GPU
- DeepSeek V3: **EP16 / DP8 / TP2**
- Llama 3.1 405B: **TP8 / DP2**

## 설치 & 활성화

vLLM은 멀티 GPU(EP/DP/TP)에서 모델 forward를 **워커 서브프로세스**에서 실행하므로,
프로파일러가 **각 워커 안에서** 모델 빌드 전에 패치돼야 합니다. 이를 위해 vLLM의
general-plugin 메커니즘(`vllm.general_plugins`, 모든 워커/엔진 프로세스에서 로드)에
등록합니다 — `vllm serve` 포함 모든 실행 방식에서 동일하게 동작합니다.

```bash
git clone https://github.com/tu-kim/vLLM-profile.git
cd vLLM-profile
pip install -e .          # vllm.general_plugins entry point 등록
```

설치 후엔 **환경변수만**으로 켜집니다 (코드 수정 불필요):

```bash
# vllm serve (권장 경로)
VLLM_PROFILER=moe,attn vllm serve deepseek-ai/DeepSeek-V3 --enforce-eager ...

# openai api_server 도 동일
VLLM_PROFILER=moe,attn python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V3 --enforce-eager ...

# 결과 집계
python -m vllm_profiler.summarize ./vllm_prof_out
```

- `VLLM_PROFILER`가 미설정/`off`면 설치돼 있어도 아무 패치도 하지 않습니다 (always-on 아님).
- 항목별로 따로: `VLLM_PROFILER=moe` 또는 `VLLM_PROFILER=attn`.

### in-process `LLM` API 또는 plugin 없이 쓰기
직접 import해서 켤 수도 있습니다 (단일 프로세스/직접 제어 시):

```python
import vllm_profiler
vllm_profiler.enable("moe", "attn")   # enable("moe") / enable("attn") 개별 토글
```

> **`pip install` 없이 빠르게:** site-packages에 `import vllm_profiler`를 실행하는
> `.pth` 파일을 두면 모든 파이썬 프로세스에서 자동 import됩니다. 단, plugin 방식이
> 워커 로드 타이밍·이식성 면에서 더 안전합니다.

> **주의:** vLLM은 기본적으로 CUDA Graph / `torch.compile`을 사용하며 이 구간에선 Python
> 레벨 forward 훅이 우회될 수 있습니다. 정확한 계측을 위해 프로파일링 실행은
> `--enforce-eager`로 돌리세요.

## 측정 항목

**MoE** (`FusedMoEKernelModularImpl` hook)
1. 토큰별 Expert Dispatch/Combine 전송 사이즈
2. 토큰 전송 방식 (배치 묶음 vs 개별 토큰)
3. Expert 부하 분포 (토큰별 routing 분포)

**Attention** (`LlamaAttention` / `MultiHeadLatentAttentionWrapper` hook)
1. GQA vs MLA 구조에 따른 시간 차이
2. 연산 / 통신 시간 breakdown
3. 메모리 load/store 오버헤드 (유효 대역폭)

자세한 내용·필드 정의는 [`vllm_profiler/README.md`](vllm_profiler/README.md) 참고.

## 구조

```
vllm_profiler/
├── __init__.py     # enable(...) 진입점 + VLLM_PROFILER 환경변수
├── recorder.py     # rank별 JSONL 기록 (버퍼 + atexit flush)
├── timing.py       # CUDA event deferred 타이머 (flush 시 sync 1회)
├── moe_hooks.py    # MoE dispatch/combine/load-balance
├── attn_hooks.py   # MLA vs GQA 연산/통신/메모리 breakdown
├── summarize.py    # 결과 집계 CLI
└── README.md
```
