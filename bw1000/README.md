# BW1000 指令耗时

1、测试设备为 BW1000 / gfx936。**每个耗时单元格均为 `timer tick / ns`**；1 tick约0.667 ns

2、我们的计时方式有如下几种。假设一条指令分为发出请求、执行操作、结果可用、观察到指令完成4个阶段
- **Dependent latency**：一条指令从发出请求到结果可用的时间
- **Completion**：一条指令从发出请求到观察到指令完成的时间
- **Issue interval**：独立指令的发射间隔

## 1. MMAC

单位为每条native MMAC；acc1为依赖链，4个独立accumulator已达到issue平台。

| Instruction | Dependent / acc1 | acc2 | Issue / acc4 | acc8 |
|---|---|---|---|---|
| `v_mmac_f32_16x16x16_bf16` | 19.99 / 13.33 | 11.99 / 8.00 | 8.00 / 5.34 | 8.00 / 5.34 |
| `v_mmac_i32_16x16x32_i8` | 19.99 / 13.33 | 11.99 / 8.00 | 8.00 / 5.34 | 8.00 / 5.34 |

## 2. Wave内跨Lane交换

| Instruction | Dependent latency | Issue interval |
|---|---|---|
| `v_mov_b32_dpp` | 4.47 / 2.98 | 4.00 / 2.67 |
| `ds_swizzle_b32` | 47.42 / 31.63 | 4.63 / 3.09 |
| `ds_bpermute_b32` | 55.42–56.41 / 36.97–37.62 | 8.05 / 5.37 |
| `ds_permute_b32` | 55.42–56.36 / 36.97–37.59 | 8.04 / 5.36 |


## 3. Lane-local / Packed ALU

### 3.1 原生指令与两条scalar组合

`2x`项按两条指令时长取平均，完整的两条指令耗时为表值的2倍，Issue时有两条独立的依赖链

| Instruction / Pair | Dependent latency | Issue interval |
|---|---|---|
| `v_perm_b32` | 4.73 / 3.15 | 4.75–4.79 / 3.17–3.20 |
| `v_pk_add_f16` | 4.20 / 2.80 | 4.75 / 3.17 |
| `v_pk_mul_f16` | 5.23 / 3.49 | 3.75 / 2.50 |
| `v_pk_fma_f16` | 5.23 / 3.49 | 4.01 / 2.67 |
| `v_pk_add_f32` | 3.02 / 2.01 | 4.51 / 3.01 |
| `2xv_add_f16` | 4.00 / 2.67 | 4.01 / 2.67 |
| `2xv_mul_f16` | 4.00 / 2.67 | 4.00 / 2.67 |
| `2xv_fma_f16` | 8.06 / 5.38 | 8.03 / 5.36 |

### 3.2 转换、组合与BF16 lowering

| Sequence | Dependent sequence cost | Issue sequence cost | Issue S |
|---|---|---|---|
| `compiler_bf16_add_sequence` | 180.57 / 120.45 | 180.06 / 120.11 | 2 |
| `compiler_bf16_convert_sequence` | 74.54 / 49.72 | 75.89 / 50.62 | 2 |
| `compiler_bf16_fma_sequence` | 176.91 / 118.00 | 175.54 / 117.09 | 4 |
| `compiler_bf16_mul_sequence` | 179.83 / 119.95 | 180.20 / 120.20 | 4 |
| `compiler_bf16_packed_container_fma_sequence` | 184.50 / 123.07 | 184.44 / 123.03 | 4 |
| `compiler_bf16_packed_container_mul_sequence` | 184.25 / 122.90 | 184.12 / 122.81 | 4 |
| `compiler_bf16_packed_container_sequence` | 185.11 / 123.47 | 183.43 / 122.35 | 2 |
| `pack_packed_alu_unpack` | 20.30 / 13.54 | — | — |
| `v_cvt_f16_f32+v_cvt_f32_f16` | 8.11 / 5.41 | 8.29 / 5.53 | 2 |
| `v_pack_b32_f16+2xcvt` | 12.30 / 8.21 | 12.15 / 8.10 | 2 |
| `v_perm+v_pk_add+and+shift` | 20.22 / 13.49 | 17.03 / 11.36 | 2 |
| `v_perm_convert_v_perm_pk_add` | 20.35 / 13.57 | — | — |

## 4. LDS核心指令


| Instruction | Completion | Dependent latency | Initial issue interval |
|---|---|---|---|
| `ds_read_b32` | 28.00 / 18.68 | 33.81 / 22.56 | 16.00 / 10.67 |
| `ds_write_b32` | 40.00 / 26.68 | — | 24.00 / 16.01 |
| `ds_read_b64` | 36.00 / 24.01 | 55.80 / 37.22 | 16.00 / 10.67 |
| `ds_write_b64` | 48.00 / 32.02 | — | 32.00 / 21.35 |
| `ds_read_b96` | 92.00 / 61.37 | 101.80 / 67.90 | 16.00 / 10.67 |
| `ds_write_b96` | 116.00 / 77.38 | — | 36.00 / 24.01 |
| `ds_read_b128` | 40.00 / 26.68 | 49.88 / 33.27 | 16.00 / 10.67 |
| `ds_write_b128` | 68.00 / 45.36 | — | 44.00 / 29.35 |

### 4.1 Bank conflict

Conflict为测试地址模式参数；conflict=1见上表。

| Instruction | Conflict | Completion | Dependent latency | Initial issue interval |
|---|---|---|---|---|
| `ds_read_b32` | 2 | 32.00 / 21.35 | 37.81 / 25.22 | 16.00 / 10.67 |
| `ds_read_b32` | 4 | 36.00 / 24.01 | 42.13 / 28.10 | 16.00 / 10.67 |
| `ds_read_b32` | 8 | 44.00 / 29.35 | 50.14 / 33.44 | 16.00 / 10.67 |
| `ds_write_b32` | 2 | 40.00 / 26.68 | — | 24.00 / 16.01 |
| `ds_write_b32` | 4 | 44.00 / 29.35 | — | 24.00 / 16.01 |
| `ds_write_b32` | 8 | 52.00 / 34.69 | — | 24.00 / 16.01 |
| `ds_read_b64` | 2 | 38.00 / 25.35 | 59.88 / 39.94 | 16.00 / 10.67 |
| `ds_read_b64` | 4 | 44.00 / 29.35 | 69.66 / 46.47 | 16.00 / 10.67 |
| `ds_read_b64` | 8 | 62.00 / 41.36 | 82.15 / 54.80 | 16.00 / 10.67 |
| `ds_write_b64` | 2 | 52.00 / 34.69 | — | 32.00 / 21.35 |
| `ds_write_b64` | 4 | 60.00 / 40.02 | — | 32.00 / 21.35 |
| `ds_write_b64` | 8 | 76.00 / 50.69 | — | 32.00 / 21.35 |
| `ds_read_b96` | 2 | 92.00 / 61.37 | 101.79 / 67.90 | 16.00 / 10.67 |
| `ds_read_b96` | 4 | 92.00 / 61.37 | 102.12 / 68.12 | 16.00 / 10.67 |
| `ds_read_b96` | 8 | 92.00 / 61.37 | 102.12 / 68.12 | 16.00 / 10.67 |
| `ds_write_b96` | 2 | 116.00 / 77.38 | — | 36.00 / 24.01 |
| `ds_write_b96` | 4 | 116.00 / 77.38 | — | 36.00 / 24.01 |
| `ds_write_b96` | 8 | 116.00 / 77.38 | — | 36.00 / 24.01 |
| `ds_read_b128` | 2 | 44.00 / 29.35 | 53.81 / 35.89 | 16.00 / 10.67 |
| `ds_read_b128` | 4 | 60.00 / 40.02 | 70.10 / 46.76 | 16.00 / 10.67 |
| `ds_read_b128` | 8 | 92.00 / 61.37 | 102.11 / 68.11 | 16.00 / 10.67 |
| `ds_write_b128` | 2 | 76.00 / 50.69 | — | 44.00 / 29.35 |
| `ds_write_b128` | 4 | 92.00 / 61.37 | — | 44.00 / 29.35 |
| `ds_write_b128` | 8 | 124.00 / 82.71 | — | 44.00 / 29.35 |

## 5. LDS扩展指令

| Instruction | Bytes/lane | Completion | Initial issue interval |
|---|---|---|---|
| `ds_read2_b32` | 8 | 52.00 / 34.69 | 16.00 / 10.67 |
| `ds_read2_b64` | 16 | 60.00 / 40.02 | 16.00–186.00 / 10.67–124.07 |
| `ds_read2st64_b32` | 8 | 48.00 / 32.02 | 158.00 / 105.39 |
| `ds_read2st64_b64` | 16 | 60.00 / 40.02 | 16.00 / 10.67 |
| `ds_read_i16` | 2 | 44.00 / 29.35 | 118.00–152.00 / 78.71–101.39 |
| `ds_read_i8` | 1 | 44.00 / 29.35 | 154.00 / 102.72 |
| `ds_read_u16` | 2 | 44.00 / 29.35 | 156.00 / 104.06 |
| `ds_read_u8` | 1 | 44.00 / 29.35 | 128.00 / 85.38 |
| `ds_write2_b32` | 8 | 64.00 / 42.69 | 28.00 / 18.68 |
| `ds_write2_b64` | 16 | 128.00 / 85.38 | 60.00–194.00 / 40.02–129.40 |
| `ds_write2st64_b32` | 8 | 60.00 / 40.02 | 28.00 / 18.68 |
| `ds_write2st64_b64` | 16 | 120.00 / 80.04 | 188.00 / 125.40 |
| `ds_write_b16` | 2 | 52.00 / 34.69 | 132.00–176.00 / 88.05–117.40 |
| `ds_write_b8` | 1 | 52.00 / 34.69 | 140.00–158.00 / 93.39–105.39 |

## 6. LDS → Global组合

| Sequence | 状态 | 整条链 | 每byte |
|---|---|---|---|
| `ds_read_b32 + global_store_dword` | L2-hot | 2116.00 / 1411.45 | 0.52 / 0.34 |
| `ds_read_b32 + global_store_dword` | L2-cold | 2126.00 / 1418.12 | 0.52 / 0.35 |
| `ds_read_b32 + global_store_dword` | streaming | 2116.00 / 1411.43 | 0.52 / 0.34 |
| `ds_read_b64 + global_store_dwordx2` | L2-hot | 2262.00 / 1508.81 | 0.28 / 0.18 |
| `ds_read_b64 + global_store_dwordx2` | L2-cold | 2264.00 / 1510.15 | 0.28 / 0.18 |
| `ds_read_b64 + global_store_dwordx2` | streaming | 2250.00 / 1500.81 | 0.27 / 0.18 |
| `ds_read_b128 + global_store_dwordx4` | L2-hot | 2492.00 / 1662.25 | 0.15 / 0.10 |
| `ds_read_b128 + global_store_dwordx4` | L2-cold | 2490.00 / 1660.92 | 0.15 / 0.10 |
| `ds_read_b128 + global_store_dwordx4` | streaming | 2490.00 / 1660.89 | 0.15 / 0.10 |

## 7. Global Load

| Instruction | L1 dependent | L2 dependent | HBM dependent | Issue interval |
|---|---|---|---|---|
| `global_load_ubyte` | — | — | — | 5.58 / 3.72 |
| `global_load_ushort` | 143.55 / 95.75 | 273.23 / 182.25 | 524.01 / 349.53 | 5.58 / 3.72 |
| `global_load_dword` | 116.01 / 77.38 | 247.37 / 165.01 | 468.68 / 312.62 | 4.00 / 2.67 |
| `global_load_dwordx2` | 132.00 / 88.05 | 261.37 / 174.34 | 484.30 / 323.05 | 5.89 / 3.93 |
| `global_load_dwordx3` | 128.00 / 85.38 | 258.38 / 172.35 | 480.21 / 320.31 | 5.94 / 3.96 |
| `global_load_dwordx4` | 132.01 / 88.05 | 262.43 / 175.05 | 485.39 / 323.77 | 5.79 / 3.86 |

### 7.1 独立请求Completion

| Instruction | L2-hot completion | L2-cold completion | HBM completion |
|---|---|---|---|
| `global_load_ubyte` | 490.00 / 326.85 | 568.00 / 378.88 | 568.00 / 378.88 |
| `global_load_ushort` | 494.00 / 329.51 | 578.00 / 385.55 | 572.00 / 381.54 |
| `global_load_dword` | 492.00 / 328.18 | 576.00 / 384.21 | 580.00 / 386.88 |
| `global_load_dwordx3` | 524.00 / 349.53 | 616.00 / 410.89 | 608.00 / 405.56 |
| `global_store_byte` | 430.00 / 286.82 | 408.00 / 272.15 | 418.00 / 278.82 |
| `global_store_short` | 428.00 / 285.49 | 416.00 / 277.49 | 426.00 / 284.16 |
| `global_store_dword` | 438.00 / 292.16 | 426.00 / 284.15 | 426.00 / 284.15 |
| `global_store_dwordx3` | 496.00 / 330.85 | 484.00 / 322.84 | 486.00 / 324.18 |

### 7.2 扩展宽度独立链

| Instruction | Hot dependent / store completion | Cold dependent / store completion | Hot issue | Cold issue |
|---|---|---|---|---|
| `global_load_ubyte` | 523.87 / 349.72 | 524.05 / 349.56 | 5.50 / 3.67 | 5.50 / 3.67 |
| `global_load_ushort` | 503.73 / 336.01 | 506.55 / 337.88 | 5.50 / 3.67 | 5.50 / 3.67 |
| `global_load_dwordx3` | 524.63 / 350.19 | 524.77 / 350.33 | 6.13 / 4.09 | 6.13 / 4.09 |
| `global_store_byte` | 316.02 / 210.97 | 330.02 / 220.31 | 8.50 / 5.67 | 8.50 / 5.67 |
| `global_store_short` | 332.02 / 221.66 | 346.01 / 230.80 | 8.50 / 5.67 | 8.50 / 5.67 |
| `global_store_dwordx3` | 376.02 / 251.04 | 386.02 / 257.49 | 19.75 / 13.18 | 19.75 / 13.18 |

## 8. Global Store

| Instruction | Issue interval / request | N32 drain / batch |
|---|---|---|
| `global_store_byte` | 7.15 / 4.77 | 352.00 / 234.79 |
| `global_store_short` | 7.14 / 4.77 | 374.00 / 249.47 |
| `global_store_dword` | 7.43 / 4.96 | 386.77 / 257.99 |
| `global_store_dwordx2` | 15.15 / 10.10 | 520.00 / 346.85 |
| `global_store_dwordx3` | 20.00 / 13.34 | 452.04 / 301.52 |
| `global_store_dwordx4` | 20.00 / 13.34 | 399.32 / 266.36 |

| Instruction | HBM streaming GB/s（512MiB） |
|---|---|
| `global_store_dword` | 1134.90 |
| `global_store_dwordx2` | 1011.27 |
| `global_store_dwordx4` | 935.65 |

## 9. Buffer Load

Offen；L1/HBM default、L2 glc。Byte/short为delta-chain，dword及以上为exact pointer chase；issue为N4/8/16拟合。

| Instruction | L1 dependent | L2 dependent | HBM dependent | Issue interval |
|---|---|---|---|---|
| `buffer_load_ubyte` | 127.56 / 85.09 | 257.92 / 172.04 | 480.47 / 320.49 | 5.00 / 3.33 |
| `buffer_load_ushort` | 127.56 / 85.09 | 257.29 / 171.62 | 489.47 / 326.49 | 5.00 / 3.33 |
| `buffer_load_dword` | 96.00 / 64.04 | 226.38 / 151.01 | 447.59 / 298.56 | 4.00 / 2.67 |
| `buffer_load_dwordx2` | 128.01 / 85.39 | 257.37 / 171.67 | 480.19 / 320.30 | 5.89 / 3.93 |
| `buffer_load_dwordx3` | 108.01 / 72.04 | 238.38 / 159.01 | 460.27 / 307.02 | 5.68 / 3.79 |
| `buffer_load_dwordx4` | 112.00 / 74.71 | 242.39 / 161.68 | 466.46 / 311.14 | 5.89 / 3.93 |

## 10. Buffer Store

L2-hot。Issue为N4/8/16拟合；N32 total按每request，drain按整个N32 batch计时。

| Instruction | Issue interval / request | N32 total / request | N32 drain / batch |
|---|---|---|---|
| `buffer_store_byte` | 6.51 / 4.34 | 17.37 / 11.59 | 372.00 / 248.13 |
| `buffer_store_short` | 4.00 / 2.67 | 17.38 / 11.59 | 388.00 / 258.81 |
| `buffer_store_dword` | 3.57 / 2.38 | 18.04 / 12.04 | 425.28 / 283.68 |
| `buffer_store_dwordx2` | 8.00 / 5.34 | 28.56 / 19.05 | 559.98 / 373.53 |
| `buffer_store_dwordx3` | 17.08 / 11.39 | 30.50 / 20.35 | 452.00 / 301.50 |
| `buffer_store_dwordx4` | 16.00 / 10.67 | 29.08 / 19.40 | 418.59 / 279.21 |

## 11. 同步与等待

### 11.1 Barrier

| Instruction | Waves | 每次barrier |
|---|---|---|
| `s_barrier` | 1 | 4.00 / 2.67 |
| `s_barrier` | 2 | 4.00 / 2.67 |
| `s_barrier` | 4 | 4.00 / 2.67 |
| `s_barrier` | 8 | 12.06 / 8.05 |

### 11.2 Empty wait


| Instruction | Matched single | Event-only / tick等值与ns |
|---|---|---|
| `s_waitcnt lgkmcnt(0)` | 4.00 / 2.67 | 4.00 / 2.67 |
| `s_waitcnt lgkmcnt(1)` | 4.00 / 2.67 | 3.99 / 2.66 |
| `s_waitcnt lgkmcnt(2)` | 4.00 / 2.67 | 3.99 / 2.66 |
| `s_waitcnt lgkmcnt(15)` | 4.00 / 2.67 | 3.98 / 2.66 |
| `s_waitcnt vmcnt(0)` | 4.00 / 2.67 | 4.00 / 2.67 |
| `s_waitcnt vmcnt(32)` | 4.00 / 2.67 | 4.00 / 2.67 |

### 11.3 Wait观察区间

| Instruction | 请求 | N | 状态 | Observed wait | Completion chain |
|---|---|---|---|---|---|
| `s_waitcnt lgkmcnt(0)` | read | 8 | none | 40.00 / 26.68 | 292.00 / 194.77 |
| `s_waitcnt lgkmcnt(15)` | read | 8 | none | 8.00 / 5.34 | 176.00–206.00 / 117.50–137.41 |
| `s_waitcnt lgkmcnt(0)` | write | 8 | none | 52.00 / 34.69 | 656.00 / 437.58 |
| `s_waitcnt lgkmcnt(15)` | write | 8 | none | 8.00 / 5.34 | 548.00 / 365.53 |
| `s_waitcnt vmcnt(15)` | read | 0 | L2-hot | 8.00 / 5.34 | 20.00 / 13.34 |
| `s_waitcnt vmcnt(0)` | read | 8 | HBM-like | 1120.00 / 747.72 | 1172.00 / 782.44 |
| `s_waitcnt vmcnt(0)` | read | 8 | L2-hot | 528.00 / 352.19 | 580.00 / 386.88 |

### 11.4 混合队列

同时有LDS和VMEM请求；表为观察到的wait区间及请求+wait整段。

| Instruction | L2-hot wait | L2-hot completion | HBM-like wait | HBM-like completion |
|---|---|---|---|---|
| `s_waitcnt lgkmcnt(0)` | 48.00 / 32.02 | 580.00 / 386.88 | 48.00 / 32.02 | 1158.00 / 772.43 |
| `s_waitcnt vmcnt(0)` | 484.00 / 322.84 | 634.00 / 422.89 | 996.00 / 664.37 | 1210.00 / 807.11 |
| `s_waitcnt vmcnt(0) lgkmcnt(0)` | 486.00 / 324.18 | 646.00 / 430.90 | 978.00–1086.00 / 652.36–724.40 | 1198.00 / 799.10 |
| `s_waitcnt 0` | 476.00 / 317.51 | 628.00 / 418.90 | 900.00–1010.00 / 600.33–673.71 | 1226.00 / 817.78 |

### 11.5 LDS + barrier交换序列

2 waves，4B/lane，payload=1，conflict=1，skew=0；整条交换序列延迟。

| Pattern | 整条exchange | Barrier span |
|---|---|---|
| pairwise | 442.00 / 294.83 | 232.00 / 154.75 |
| reduction | 908.00 / 606.19 | 12.00 / 8.00 |
| layout | 556.00 / 371.19 | 20.00 / 13.34 |

### 11.6 VALU依赖延时序列

每次迭代为`2×v_add_u32 + 2×v_xor_b32`；表为整个delay段，含loop和计时开销。

| 迭代次数 | 整个delay段 |
|---|---|
| 0 | 68.00 / 45.36 |
| 4 | 260.00 / 173.43 |
| 8 | 484.00 / 322.84 |
| 16 | 932.00 / 621.68 |
| 32 | 1828.00 / 1219.34 |
| 64 | 3620.00 / 2414.67 |
| 128 | 7204.00 / 4805.32 |
| 256 | 14372.00 / 9586.62 |
