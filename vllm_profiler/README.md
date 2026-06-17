# vllm_profiler — vLLM 0.18.0 내부 동작 프로파일링 addon

DeepSeek V3 (MLA + fine-grained MoE)와 Llama 3.1 405B (GQA + dense MLP)의 내부
동작을 **vLLM 소스 수정 없이** monkey-patch로 계측합니다.

- 대상 환경: H100 × 8 × 2대 = 16 GPU
  - DeepSeek: **EP16 / DP8 / TP2**
  - Llama: **TP8 / DP2**
- 항목별로 따로 켤 수 있음 (`"moe"` / `"attn"`) — 한꺼번에 다 돌릴 필요 없음.

## 빠른 시작

### 1) 코드로 활성화
vLLM이 import 가능해진 직후(엔진 생성 전/직후) 한 번 호출:

```python
import vllm_profiler
vllm_profiler.enable("moe")          # MoE만
vllm_profiler.enable("attn")         # Attention만
vllm_profiler.enable("moe", "attn")  # 둘 다
```

### 2) `vllm serve` / api_server (plugin 등록, 권장)
멀티 GPU는 모델이 워커 서브프로세스에서 돌기 때문에, 각 워커에서 패치되도록 vLLM
general-plugin으로 등록합니다. `pip install -e .` 한 번이면 entry point가 잡히고,
이후엔 환경변수만으로 켜집니다 (`vllm serve`에서도 동일하게 동작):

```bash
pip install -e .    # vllm.general_plugins entry point 등록 (plugin.py:register)
VLLM_PROFILER=moe,attn  vllm serve deepseek-ai/DeepSeek-V3 --enforce-eager ...
```

`register()`는 `worker_base.py`의 `load_general_plugins()`가 **각 워커**에서 모델
빌드 전에 호출하므로 EP/DP/TP fan-out 전체에 패치가 적용됩니다. `VLLM_PROFILER`가
미설정이면 설치돼 있어도 패치하지 않습니다.

> **중요 — `--enforce-eager` 권장.** vLLM은 CUDA Graph / `torch.compile`을 기본
> 사용합니다. 컴파일/그래프 캡처 구간에서는 Python 레벨 forward 훅이 우회되거나
> graph break를 유발할 수 있습니다. 정확한 내부 계측을 위해 프로파일링 실행은
> `--enforce-eager`(eager 모드)로 돌리세요.

각 rank가 `./vllm_prof_out/prof_rank{R}.jsonl`에 기록합니다
(`VLLM_PROFILER_DIR`로 경로 변경).

### 3) 결과 요약
```bash
python -m vllm_profiler.summarize ./vllm_prof_out             # 기본: 앞 30000줄/파일 스킵 후 평균
python -m vllm_profiler.summarize ./vllm_prof_out --skip=0          # 스킵 없이 전체
python -m vllm_profiler.summarize ./vllm_prof_out --skip=50000      # 앞 50000줄 제외
python -m vllm_profiler.summarize ./vllm_prof_out --phase=decode    # decode만
python -m vllm_profiler.summarize ./vllm_prof_out --phase=prefill   # prefill만
```

**init/warmup 제거 = 앞쪽 고정 스킵.** vLLM 초기화(메모리 프로파일링, DeepGEMM/FlashInfer
warmup, Triton/샘플러 warmup, cudagraph 캡처)는 각 rank 파일 **앞쪽**에 기록됩니다. 이
init 레코드를 신뢰성 있게 태깅하기 어렵고(경로가 제각각), serving-gate 방식은 일부 rank에서
오태깅 문제가 있어, 단순하게 **파일별 앞 N줄(기본 30000)을 고정 제외**하고 steady-state로
평균냅니다.
- `--skip=N` 으로 줄 수 직접 지정, `--skip=0` 으로 스킵 해제.
- 기본값은 `VLLM_PROFILER_SKIP`(기본 30000) env로 변경 가능.

**prefill/decode 라벨:** 실제 forward는 `execute_model`에서 요청별 토큰 수로 분류해
모든 이벤트에 `batch_type`(`prefill`/`decode`/`mixed`)을 태깅합니다. (chunked-prefill을
안 쓰면 `mixed`는 나오지 않습니다.) `--phase=`로 단계별 집계 가능.

**DP idle-rank dummy 필터:** 서빙 중 일감 없는 DP rank가 lockstep을 위해 돌리는 1-토큰
dummy forward는 `_dummy_run` 경로라 ① **`dummy: True`로 태깅**되고 ② `execute_model`을
안 타서 **`batch_type`이 없습니다**. summarize가 `dummy=True`를 **기본 제외**합니다
(`--include-dummy`로 포함). 추가로 `--phase=prefill`/`decode`는 batch_type 없는 dummy를
자동으로 걸러냅니다 — 두 신호 중 아무거나로 구분 가능.

## 측정 항목 → 구현 매핑

### MoE (`enable("moe")`) — hook: `FusedMoEKernelModularImpl`
| 요청 항목 | 기록 `kind` | 핵심 필드 |
|---|---|---|
| **MoE 전체 레이어 시간** | `moe_total`(ms) | `DeepseekV2MoE.forward` 전체(gate+dispatch+experts+combine+shared+all-reduce), attn_total의 대응 |
| ① Dispatch/Combine 토큰별 전송 사이즈 + **batch된 토큰 수** | `moe_dispatch_size`, `moe_dispatch_tokens`, `moe_combine_size`, `moe_dispatch`/`moe_combine`(ms) | `bytes_in`/`bytes_recv`/`bytes_out`, `per_token_bytes`, `tokens_in`, `routing_slots_sent`(=tok×topk), `tokens_recv`(Standard) 또는 `n_local_experts`/`max_tokens_per_expert`/`tokens_recv_padded`(BatchedExperts), `expert_num_tokens`(expert별 실제 토큰 수) |
| ② 토큰 전송 방식 (배치 묶음 vs 개별) | `moe_call` | `pf_class`, `act_format`, `grouping` |
| ②-b Sequence-parallel **청크 패딩 오버헤드** | `moe_call` (sp 필드) | `is_sequence_parallel`, `tokens_before_chunk`, `tp_size`, `padded_len`, `pad_total`, `chunk_tokens`, `pad_tokens_this_rank` |
| ③ Expert 부하 분포 (토큰별 routing) + **배치별 불균형** | `moe_expert_load` | `counts`(expert별 routing 횟수), 배치별: `cov`/`max_over_mean`(expert 단위), `rank_cov`/`rank_max_over_mean`(EP-rank 단위, 가장 느린 rank가 step을 좌우) |

- **②의 해석**: `grouping` 값
  - `batched_grouped_by_rank` → DeepEP HT / pplx: 목적지 rank별로 토큰을 **한 버퍼에 묶어** all-to-all (배치 전송)
  - `per_token_low_latency` → DeepEP LL: 저지연 경로
  - `batched_per_expert_padded` → expert별 padded `[E, max_tokens, H]` 배치
- **③의 해석**: summarize가 layer별 `max/mean`, `CoV`(변동계수)를 출력 → 값이 클수록
  expert 불균형. EPLB 효과 측정에 사용.

### Attention (`enable("attn")`) — hook: `LlamaAttention` / `MultiHeadLatentAttentionWrapper`
| 요청 항목 | 기록 `kind` | 핵심 필드 |
|---|---|---|
| ① GQA vs MLA 구조 시간차 | `attn_total` | `attn_cls`, `ms`, `num_tokens` |
| ② 연산/통신 breakdown | `attn_step`, `tp_all_reduce` | `role`(proj/proj_down/proj_up/rope/core_attn/out_proj_comm), `ms` |
| ③ 메모리 load/store 오버헤드 | `attn_step`, `attn_step_io` | `bytes_in`/`bytes_out`, `in_shape`/`out_shape` → 유효 대역폭 = bytes/time |

- `role=out_proj_comm` = RowParallelLinear (TP all-reduce 포함). `tp_all_reduce`
  레코드가 **순수 통신 시간**을 분리해줌 → 연산 = out_proj 전체 − all_reduce.
- 메모리 오버헤드: summarize가 role별 누적 바이트 / 누적 시간으로 **유효 GB/s**를 산출.

## 동작 원리 (간단)
- **타이밍**: `torch.cuda.Event` 쌍을 기록만 하고, flush 시점에 `synchronize()`를
  **한 번만** 호출해 일괄 변환 → per-call sync로 인한 성능 왜곡 없음.
  (CUDA 없으면 `perf_counter` fallback → import/단위테스트 가능)
- **무침습**: vLLM 클래스 메서드와 서브모듈 `.forward`만 교체. 소스 파일 변경 0.
- **rank 인식**: `torch.distributed` 또는 `RANK`/`VLLM_DP_RANK`/`LOCAL_RANK` 환경변수.

## 환경변수
| 변수 | 기본값 | 설명 |
|---|---|---|
| `VLLM_PROFILER` | (없음) | `moe,attn` 등 자동 활성화 |
| `VLLM_PROFILER_DIR` | `./vllm_prof_out` | 출력 디렉터리 |
| `VLLM_PROFILER_COMM` | `1` | attn의 TP all-reduce 통신 타이밍 on/off |
| `VLLM_PROFILER_FLUSH_EVERY` | `1024` | 디스크 쓰기 주기 (레코드 N개마다). 클수록 디스크 I/O↓ (sync 없음) |
| `VLLM_PROFILER_RESOLVE_EVERY` | `2048` | CUDA event resolve 주기 (= `synchronize()` 지점). **클수록 sync 드물어져 측정 오버헤드↓** (권장). 낮추면 타이밍 ms가 더 빨리 생성되나 sync 잦아짐 |

## 비활성화
```python
vllm_profiler.disable()   # 모든 hook 제거 + flush
```
