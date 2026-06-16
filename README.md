# vLLM-profile

vLLM **0.18.0** 내부 동작 프로파일링 addon — **DeepSeek V3** (MLA + fine-grained MoE)와
**Llama 3.1 405B** (GQA + dense MLP)의 MoE / Attention 내부 동작을 **vLLM 소스 수정 없이**
monkey-patch로 계측합니다.

대상 환경: H100 × 8 × 2대 = 16 GPU
- DeepSeek V3: **EP16 / DP8 / TP2**
- Llama 3.1 405B: **TP8 / DP2**

## 설치

vLLM 0.18.0이 설치된 환경에서, 이 저장소를 PYTHONPATH에 두거나 프로젝트에 복사:

```bash
git clone https://github.com/tu-kim/vLLM-profile.git
export PYTHONPATH=$PWD/vLLM-profile:$PYTHONPATH
```

## 사용

```bash
# import 한 줄 + 환경변수로 활성화 (enforce-eager 권장)
VLLM_PROFILER=moe,attn python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V3 --enforce-eager ...

# 결과 집계
python -m vllm_profiler.summarize ./vllm_prof_out
```

또는 코드에서:

```python
import vllm_profiler
vllm_profiler.enable("moe")          # MoE만
vllm_profiler.enable("attn")         # Attention만
vllm_profiler.enable("moe", "attn")  # 둘 다 (항목별로 따로 켤 수 있음)
```

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
