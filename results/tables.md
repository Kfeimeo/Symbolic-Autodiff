## T1 Tree AST vs Tensor DAG (symbolic size, differentiation time, memory)

| configuration | DAG fwd nodes | DAG bwd nodes | DAG diff time | DAG graph bytes | tree fwd nodes | tree grad nodes | tree diff time | tree bytes |
|---|---|---|---|---|---|---|---|---|
| chain(n=1024,depth=4) | 22 | 20 | 8.9 us | 112.5 KiB | 22 | 717 | 186.7 us | 97.7 KiB |
| chain(n=1024,depth=16) | 82 | 68 | 29.8 us | 114.0 KiB | 82 | 20453 | 3.46 ms | 2.7 MiB |
| chain(n=1024,depth=64) | 322 | 260 | 101.0 us | 119.8 KiB | 322 | 977797 | 168.13 ms | 126.8 MiB |
| chain(n=1024,depth=256) | 1282 | 1028 | 1.05 ms | 479.0 KiB | 1282 | cap hit | 410.16 ms (cap hit) | - |
| sharing(n=1024,depth=8,k=1) | 42 | 36 | 22.8 us | 113.0 KiB | 42 | 3509 | 566.4 us | 470.6 KiB |
| sharing(n=1024,depth=8,k=2) | 50 | 44 | 19.6 us | 113.3 KiB | 2807 | cap hit | 382.78 ms (cap hit) | - |
| sharing(n=1024,depth=8,k=4) | 66 | 60 | 26.1 us | 113.8 KiB | 502437 | cap hit | 368.49 ms (cap hit) | - |
| diamond(n=4096,depth=4,width=2) | 30 | 40 | 15.9 us | 113.0 KiB | 122 | 11307 | 2.70 ms | 1.5 MiB |
| diamond(n=4096,depth=4,width=4) | 62 | 80 | 49.2 us | 114.0 KiB | 1532 | 1226687 | 199.24 ms | 159.3 MiB |
| diamond(n=4096,depth=4,width=8) | 126 | 160 | 66.0 us | 116.0 KiB | 22232 | cap hit | 384.29 ms (cap hit) | - |
| mlp(batch=4,in=8,hidden=8,out=8,layers=8) | 48 | 69 | 30.0 us | 114.3 KiB | 94 | 13153 | 2.61 ms | 1.8 MiB |
| mlp(batch=4,in=8,hidden=8,out=8,layers=32) | 168 | 237 | 93.4 us | 119.9 KiB | 334 | 342801 | 55.81 ms | 46.6 MiB |
| mlp(batch=4,in=8,hidden=8,out=8,layers=128) | 648 | 909 | 327.1 us | 254.4 KiB | 1294 | cap hit | 579.08 ms (cap hit) | - |

## T2 Hash-consing ablation: sharing task (depth 8)

| configuration | nodes HC / noHC | HC hits fwd+bwd | ATen ops HC / noHC / torch | fwd+bwd median HC / noHC / torch | peak live HC / noHC |
|---|---|---|---|---|---|
| sharing(n=1024,depth=8,k=1) | 78 / 85 | 0+7 | 110 / 131 / 109 | 105.1 us / 115.9 us / 358.8 us | 68.0 KiB / 68.0 KiB |
| sharing(n=1024,depth=8,k=2) | 94 / 149 | 16+7 | 126 / 195 / 197 | 131.4 us / 235.5 us / 591.3 us | 68.0 KiB / 80.0 KiB |
| sharing(n=1024,depth=8,k=4) | 126 / 277 | 48+7 | 158 / 323 / 373 | 174.7 us / 377.4 us / 1.04 ms | 68.0 KiB / 84.0 KiB |
| sharing(n=1024,depth=8,k=8) | 190 / 533 | 112+7 | 222 / 579 / 725 | 251.5 us / 747.6 us / 1.90 ms | 68.0 KiB / 84.0 KiB |
| sharing(n=65536,depth=8,k=1) | 78 / 85 | 0+7 | 116 / 137 / 115 | 2.58 ms / 2.56 ms / 2.06 ms | 4.3 MiB / 4.3 MiB |
| sharing(n=65536,depth=8,k=2) | 94 / 149 | 16+7 | 132 / 201 / 203 | 2.13 ms / 3.13 ms / 3.23 ms | 4.3 MiB / 5.0 MiB |
| sharing(n=65536,depth=8,k=4) | 126 / 277 | 48+7 | 164 / 329 / 379 | 2.53 ms / 5.59 ms / 4.57 ms | 4.3 MiB / 5.3 MiB |
| sharing(n=65536,depth=8,k=8) | 190 / 533 | 112+7 | 228 / 585 / 731 | 2.74 ms / 13.42 ms / 10.62 ms | 4.3 MiB / 5.3 MiB |

## T3 Hash-consing on tasks without syntactic duplication

| configuration | bwd nodes HC / noHC | HC hits bwd / AD requests | ATen ops HC / noHC | fwd+bwd median HC / noHC |
|---|---|---|---|---|
| chain(n=1024,depth=256) | 1028 / 1283 | 255 / 1283 | 3086 / 3851 | 5.13 ms / 5.67 ms |
| diamond(n=4096,depth=16,width=8) | 628 / 755 | 127 / 755 | 1646 / 2027 | 4.33 ms / 4.59 ms |
| nn(batch=256,in=512,out=256) | 11 / 11 | 0 / 11 | 51 / 51 | 1.12 ms / 1.06 ms |
| mlp(batch=128,in=256,hidden=256,out=64,layers=8) | 69 / 78 | 9 / 78 | 296 / 321 | 3.99 ms / 5.01 ms |
| chain(n=8,depth=4096) | 16388 / 20483 | 4095 / 20483 | 49166 / 61451 | 92.16 ms / 78.35 ms |
| diamond(n=8,depth=1024,width=4) | 19460 / 23555 | 4095 / 23555 | 51214 / 63499 | 65.60 ms / 72.39 ms |

## T4 Scaling of symbolic differentiation (hash-consing on, medians of 5 fresh graphs)

| configuration | fwd nodes | fwd edges | bwd nodes | bwd/fwd | accumulations | build | diff | diff ns / fwd node | compile |
|---|---|---|---|---|---|---|---|---|---|
| chain(n=8,depth=64) | 322 | 321 | 260 | 0.81 | 0 | 79.3 us | 99.0 us | 307 | 36.6 us |
| chain(n=8,depth=128) | 642 | 641 | 516 | 0.80 | 0 | 159.2 us | 207.0 us | 322 | 73.9 us |
| chain(n=8,depth=256) | 1282 | 1281 | 1028 | 0.80 | 0 | 597.0 us | 854.4 us | 666 | 336.2 us |
| chain(n=8,depth=512) | 2562 | 2561 | 2052 | 0.80 | 0 | 779.2 us | 883.6 us | 345 | 304.0 us |
| chain(n=8,depth=1024) | 5122 | 5121 | 4100 | 0.80 | 0 | 2.73 ms | 2.76 ms | 538 | 1.20 ms |
| chain(n=8,depth=2048) | 10242 | 10241 | 8196 | 0.80 | 0 | 5.07 ms | 8.30 ms | 810 | 4.38 ms |
| chain(n=8,depth=4096) | 20482 | 20481 | 16388 | 0.80 | 0 | 23.40 ms | 31.83 ms | 1554 | 14.80 ms |
| diamond(n=8,depth=16,width=4) | 242 | 289 | 308 | 1.27 | 48 | 70.7 us | 107.1 us | 443 | 39.1 us |
| diamond(n=8,depth=64,width=4) | 962 | 1153 | 1220 | 1.27 | 192 | 372.2 us | 717.9 us | 746 | 273.1 us |
| diamond(n=8,depth=256,width=4) | 3842 | 4609 | 4868 | 1.27 | 768 | 1.26 ms | 2.48 ms | 646 | 870.4 us |
| diamond(n=8,depth=1024,width=4) | 15362 | 18433 | 19460 | 1.27 | 3072 | 11.19 ms | 19.88 ms | 1294 | 10.06 ms |
| mlp(batch=4,in=8,hidden=8,out=8,layers=8) | 48 | 48 | 69 | 1.44 | 1 | 19.6 us | 30.0 us | 624 | 12.5 us |
| mlp(batch=4,in=8,hidden=8,out=8,layers=32) | 168 | 168 | 237 | 1.41 | 1 | 57.8 us | 93.4 us | 556 | 39.7 us |
| mlp(batch=4,in=8,hidden=8,out=8,layers=128) | 648 | 648 | 909 | 1.40 | 1 | 183.0 us | 327.1 us | 505 | 119.4 us |

## T5 Execution: DAG executor vs LibTorch Autograd (median; ratio = torch / DAG)

| configuration | ops DAG/torch | fwd DAG | fwd torch | bwd DAG | bwd torch | total DAG (P90) | total torch (P90) | fwd ratio | bwd ratio | total ratio | fwd no-grad DAG/torch | peak live DAG | peak malloc DAG/torch | grad max rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain(n=64,depth=4) | 62/61 | 11.5 us | 24.2 us | 22.6 us | 93.5 us | 34.1 us (43.6 us) | 118.6 us (162.2 us) | 2.12 | 4.14 | 3.48 | 10.8 us/10.6 us | 2.3 KiB | 2.9 KiB/8.1 KiB | 0.0e+00 |
| chain(n=64,depth=16) | 206/205 | 40.6 us | 92.4 us | 76.7 us | 386.8 us | 117.3 us (153.5 us) | 475.5 us (584.0 us) | 2.28 | 5.04 | 4.05 | 38.8 us/36.8 us | 8.3 KiB | 18.2 KiB/47.7 KiB | 0.0e+00 |
| chain(n=64,depth=64) | 782/781 | 172.7 us | 368.5 us | 302.4 us | 1.40 ms | 485.7 us (551.3 us) | 1.79 ms (1.84 ms) | 2.13 | 4.62 | 3.68 | 263.9 us/142.0 us | 32.3 KiB | 81.9 KiB/235.9 KiB | 0.0e+00 |
| chain(n=64,depth=256) | 3086/3085 | 982.5 us | 2.24 ms | 1.49 ms | 7.91 ms | 2.43 ms (3.53 ms) | 10.38 ms (12.08 ms) | 2.28 | 5.30 | 4.27 | 844.7 us/656.7 us | 128.3 KiB | 330.3 KiB/981.0 KiB | 0.0e+00 |
| chain(n=1024,depth=4) | 62/61 | 17.0 us | 34.8 us | 38.2 us | 114.5 us | 55.1 us (64.8 us) | 149.8 us (182.7 us) | 2.05 | 3.00 | 2.72 | 16.0 us/16.2 us | 36.0 KiB | 38.5 KiB/44.2 KiB | 0.0e+00 |
| chain(n=1024,depth=16) | 206/205 | 70.1 us | 162.7 us | 156.9 us | 521.7 us | 234.9 us (327.6 us) | 724.4 us (1.15 ms) | 2.32 | 3.33 | 3.08 | 59.5 us/56.8 us | 132.0 KiB | 143.7 KiB/173.3 KiB | 0.0e+00 |
| chain(n=1024,depth=64) | 782/781 | 305.1 us | 606.3 us | 667.0 us | 1.94 ms | 977.9 us (1.09 ms) | 2.54 ms (3.00 ms) | 1.99 | 2.91 | 2.60 | 272.4 us/240.1 us | 516.0 KiB | 567.5 KiB/720.8 KiB | 0.0e+00 |
| chain(n=1024,depth=256) | 3086/3085 | 1.93 ms | 3.70 ms | 3.17 ms | 11.56 ms | 5.13 ms (5.51 ms) | 15.90 ms (17.76 ms) | 1.92 | 3.64 | 3.10 | 1.11 ms/1.07 ms | 2.0 MiB | 2.2 MiB/2.8 MiB | 0.0e+00 |
| chain(n=65536,depth=4) | 68/67 | 243.4 us | 316.4 us | 666.4 us | 473.4 us | 948.4 us (1.07 ms) | 808.7 us (1.01 ms) | 1.30 | 0.71 | 0.85 | 193.5 us/227.6 us | 2.3 MiB | 2.3 MiB/2.3 MiB | 0.0e+00 |
| chain(n=65536,depth=16) | 212/211 | 2.07 ms | 1.41 ms | 4.01 ms | 2.02 ms | 6.37 ms (8.41 ms) | 3.42 ms (4.11 ms) | 0.68 | 0.50 | 0.54 | 741.1 us/707.5 us | 8.3 MiB | 8.3 MiB/8.3 MiB | 0.0e+00 |
| chain(n=65536,depth=64) | 788/787 | 8.55 ms | 7.66 ms | 15.44 ms | 10.80 ms | 24.48 ms (28.97 ms) | 18.53 ms (25.20 ms) | 0.90 | 0.70 | 0.76 | 4.56 ms/5.36 ms | 32.3 MiB | 32.3 MiB/32.4 MiB | 0.0e+00 |
| chain(n=65536,depth=256) | 3092/3091 | 34.40 ms | 26.54 ms | 60.45 ms | 43.78 ms | 98.92 ms (148.54 ms) | 72.93 ms (79.65 ms) | 0.77 | 0.72 | 0.74 | 20.79 ms/21.20 ms | 128.3 MiB | 128.4 MiB/129.1 MiB | 0.0e+00 |
| sharing(n=1024,depth=8,k=1) | 110/109 | 32.9 us | 84.5 us | 72.6 us | 273.8 us | 105.1 us (117.9 us) | 358.8 us (520.3 us) | 2.57 | 3.77 | 3.41 | 31.5 us/30.4 us | 68.0 KiB | 73.7 KiB/85.1 KiB | 0.0e+00 |
| sharing(n=1024,depth=8,k=2) | 126/197 | 43.2 us | 129.0 us | 87.6 us | 453.4 us | 131.4 us (154.7 us) | 591.3 us (722.8 us) | 2.99 | 5.18 | 4.50 | 41.3 us/56.8 us | 68.0 KiB | 73.5 KiB/113.6 KiB | 0.0e+00 |
| sharing(n=1024,depth=8,k=4) | 158/373 | 63.8 us | 240.1 us | 108.9 us | 804.6 us | 174.7 us (222.2 us) | 1.04 ms (1.39 ms) | 3.76 | 7.39 | 5.95 | 61.5 us/114.8 us | 68.0 KiB | 73.5 KiB/146.8 KiB | 0.0e+00 |
| sharing(n=1024,depth=8,k=8) | 222/725 | 102.3 us | 476.7 us | 145.7 us | 1.41 ms | 251.5 us (281.8 us) | 1.90 ms (1.97 ms) | 4.66 | 9.67 | 7.55 | 97.7 us/221.2 us | 68.0 KiB | 73.5 KiB/219.1 KiB | 1.2e-07 |
| sharing(n=65536,depth=8,k=1) | 116/115 | 735.4 us | 757.6 us | 1.78 ms | 1.28 ms | 2.58 ms (3.48 ms) | 2.06 ms (2.37 ms) | 1.03 | 0.72 | 0.80 | 421.2 us/506.7 us | 4.3 MiB | 4.3 MiB/4.3 MiB | 0.0e+00 |
| sharing(n=65536,depth=8,k=2) | 132/203 | 670.0 us | 1.16 ms | 1.40 ms | 2.03 ms | 2.13 ms (2.33 ms) | 3.23 ms (8.03 ms) | 1.73 | 1.45 | 1.51 | 499.8 us/768.8 us | 4.3 MiB | 4.3 MiB/5.0 MiB | 0.0e+00 |
| sharing(n=65536,depth=8,k=4) | 164/379 | 908.8 us | 1.69 ms | 1.53 ms | 2.83 ms | 2.53 ms (2.92 ms) | 4.57 ms (5.40 ms) | 1.86 | 1.86 | 1.81 | 694.8 us/1.28 ms | 4.3 MiB | 4.3 MiB/5.1 MiB | 0.0e+00 |
| sharing(n=65536,depth=8,k=8) | 228/731 | 1.12 ms | 4.07 ms | 1.63 ms | 6.47 ms | 2.74 ms (2.97 ms) | 10.62 ms (11.81 ms) | 3.63 | 3.97 | 3.88 | 798.7 us/4.72 ms | 4.3 MiB | 4.3 MiB/5.1 MiB | 1.2e-07 |
| diamond(n=4096,depth=4,width=2) | 110/97 | 39.2 us | 75.8 us | 152.5 us | 260.7 us | 200.3 us (276.4 us) | 336.5 us (480.9 us) | 1.93 | 1.71 | 1.68 | 34.5 us/51.0 us | 224.0 KiB | 228.6 KiB/240.8 KiB | 0.0e+00 |
| diamond(n=4096,depth=4,width=4) | 214/193 | 143.4 us | 181.1 us | 518.0 us | 499.6 us | 661.5 us (744.1 us) | 656.9 us (1.21 ms) | 1.26 | 0.96 | 0.99 | 118.0 us/73.3 us | 352.0 KiB | 359.7 KiB/386.0 KiB | 0.0e+00 |
| diamond(n=4096,depth=4,width=8) | 422/385 | 204.9 us | 352.4 us | 645.0 us | 1.00 ms | 858.3 us (936.3 us) | 1.35 ms (1.90 ms) | 1.72 | 1.55 | 1.58 | 161.6 us/153.7 us | 608.0 KiB | 621.7 KiB/685.8 KiB | 0.0e+00 |
| diamond(n=4096,depth=16,width=2) | 398/349 | 201.1 us | 321.1 us | 656.6 us | 882.7 us | 883.7 us (1.22 ms) | 1.19 ms (1.43 ms) | 1.60 | 1.34 | 1.35 | 138.3 us/131.4 us | 800.0 KiB | 818.4 KiB/876.3 KiB | 0.0e+00 |
| diamond(n=4096,depth=16,width=4) | 814/733 | 515.5 us | 691.3 us | 1.35 ms | 1.97 ms | 1.89 ms (3.18 ms) | 2.67 ms (2.89 ms) | 1.34 | 1.46 | 1.41 | 371.9 us/310.2 us | 1.3 MiB | 1.3 MiB/1.4 MiB | 0.0e+00 |
| diamond(n=4096,depth=16,width=8) | 1646/1501 | 1.23 ms | 2.06 ms | 3.03 ms | 4.82 ms | 4.33 ms (4.58 ms) | 6.87 ms (7.33 ms) | 1.68 | 1.59 | 1.59 | 751.6 us/724.4 us | 2.3 MiB | 2.3 MiB/2.6 MiB | 0.0e+00 |
| nn(batch=32,in=64,out=32) | 45/44 | 9.8 us | 20.3 us | 27.6 us | 63.6 us | 37.4 us (38.9 us) | 84.2 us (113.7 us) | 2.07 | 2.31 | 2.25 | 8.9 us/9.1 us | 28.1 KiB | 20.1 KiB/24.2 KiB | 0.0e+00 |
| nn(batch=256,in=512,out=256) | 51/50 | 363.8 us | 405.9 us | 754.0 us | 727.7 us | 1.12 ms (1.42 ms) | 1.13 ms (1.23 ms) | 1.12 | 0.97 | 1.01 | 336.3 us/294.0 us | 1.8 MiB | 1.3 MiB/1.3 MiB | 0.0e+00 |
| nn(batch=1024,in=1024,out=1024) | 51/50 | 6.90 ms | 8.84 ms | 13.28 ms | 16.48 ms | 20.35 ms (23.86 ms) | 25.29 ms (27.78 ms) | 1.28 | 1.24 | 1.24 | 5.90 ms/6.73 ms | 16.0 MiB | 12.0 MiB/12.0 MiB | 0.0e+00 |
| mlp(batch=128,in=64,hidden=64,out=64,layers=2) | 110/107 | 94.1 us | 118.7 us | 232.4 us | 315.6 us | 338.2 us (411.8 us) | 449.3 us (548.9 us) | 1.26 | 1.36 | 1.33 | 78.0 us/77.3 us | 224.0 KiB | 226.1 KiB/167.5 KiB | 0.0e+00 |
| mlp(batch=128,in=64,hidden=64,out=64,layers=4) | 172/165 | 151.6 us | 228.1 us | 396.7 us | 653.7 us | 579.0 us (684.4 us) | 891.6 us (965.8 us) | 1.50 | 1.65 | 1.54 | 142.0 us/125.0 us | 352.0 KiB | 354.9 KiB/237.4 KiB | 0.0e+00 |
| mlp(batch=128,in=64,hidden=64,out=64,layers=8) | 296/281 | 329.6 us | 357.1 us | 828.3 us | 983.1 us | 1.17 ms (1.76 ms) | 1.33 ms (1.87 ms) | 1.08 | 1.19 | 1.14 | 385.9 us/290.2 us | 608.0 KiB | 614.0 KiB/375.3 KiB | 0.0e+00 |
| mlp(batch=128,in=256,hidden=256,out=64,layers=2) | 110/107 | 437.8 us | 489.3 us | 866.0 us | 921.2 us | 1.47 ms (2.07 ms) | 1.42 ms (1.74 ms) | 1.12 | 1.06 | 0.96 | 369.6 us/404.2 us | 962.3 KiB | 834.2 KiB/839.5 KiB | 0.0e+00 |
| mlp(batch=128,in=256,hidden=256,out=64,layers=4) | 172/165 | 888.8 us | 797.4 us | 2.02 ms | 1.43 ms | 2.90 ms (3.45 ms) | 2.23 ms (2.95 ms) | 0.90 | 0.71 | 0.77 | 775.5 us/651.9 us | 1.4 MiB | 1.3 MiB/1.3 MiB | 0.0e+00 |
| mlp(batch=128,in=256,hidden=256,out=64,layers=8) | 296/281 | 1.32 ms | 1.93 ms | 2.67 ms | 3.46 ms | 3.99 ms (4.35 ms) | 5.42 ms (5.94 ms) | 1.46 | 1.30 | 1.36 | 1.25 ms/1.49 ms | 2.4 MiB | 2.3 MiB/2.3 MiB | 0.0e+00 |
| mlp(batch=128,in=1024,hidden=1024,out=64,layers=2) | 110/107 | 3.37 ms | 3.51 ms | 6.07 ms | 5.96 ms | 9.45 ms (17.01 ms) | 9.33 ms (9.95 ms) | 1.04 | 0.98 | 0.99 | 3.15 ms/3.29 ms | 9.8 MiB | 9.3 MiB/9.3 MiB | 0.0e+00 |
| mlp(batch=128,in=1024,hidden=1024,out=64,layers=4) | 172/165 | 6.39 ms | 6.57 ms | 11.48 ms | 11.00 ms | 17.96 ms (20.63 ms) | 18.06 ms (21.32 ms) | 1.03 | 0.96 | 1.01 | 6.17 ms/5.27 ms | 17.8 MiB | 17.3 MiB/17.3 MiB | 0.0e+00 |
| mlp(batch=128,in=1024,hidden=1024,out=64,layers=8) | 296/281 | 11.46 ms | 10.80 ms | 20.75 ms | 17.35 ms | 31.73 ms (42.94 ms) | 28.10 ms (38.23 ms) | 0.94 | 0.84 | 0.89 | 11.74 ms/9.83 ms | 33.8 MiB | 33.3 MiB/33.3 MiB | 0.0e+00 |
| chain(n=8,depth=64) | 782/781 | 192.3 us | 384.4 us | 346.0 us | 1.42 ms | 542.8 us (621.2 us) | 1.80 ms (1.89 ms) | 2.00 | 4.11 | 3.32 | 153.9 us/162.9 us | 4.0 KiB | 57.1 KiB/219.6 KiB | 0.0e+00 |
| chain(n=8,depth=128) | 1550/1549 | 367.6 us | 747.7 us | 679.3 us | 2.91 ms | 1.05 ms (1.13 ms) | 3.65 ms (3.75 ms) | 2.03 | 4.29 | 3.49 | 306.0 us/335.8 us | 8.0 KiB | 112.5 KiB/451.0 KiB | 0.0e+00 |
| chain(n=8,depth=256) | 3086/3085 | 1.98 ms | 1.73 ms | 3.18 ms | 6.22 ms | 5.23 ms (5.57 ms) | 7.94 ms (8.22 ms) | 0.88 | 1.96 | 1.52 | 1.25 ms/702.3 us | 16.0 KiB | 222.9 KiB/916.9 KiB | 0.0e+00 |
| chain(n=8,depth=512) | 6158/6157 | 2.07 ms | 4.12 ms | 3.26 ms | 14.71 ms | 5.32 ms (5.71 ms) | 18.73 ms (20.31 ms) | 1.99 | 4.51 | 3.52 | 1.38 ms/1.51 ms | 32.0 KiB | 443.4 KiB/1.8 MiB | 0.0e+00 |
| chain(n=8,depth=1024) | 12302/12301 | 5.26 ms | 10.45 ms | 7.99 ms | 40.14 ms | 13.25 ms (17.16 ms) | 50.27 ms (54.69 ms) | 1.99 | 5.02 | 3.80 | 3.08 ms/3.15 ms | 64.0 KiB | 884.5 KiB/3.6 MiB | 0.0e+00 |
| chain(n=8,depth=2048) | 24590/24589 | 11.77 ms | 24.42 ms | 18.56 ms | 91.89 ms | 31.84 ms (47.34 ms) | 119.24 ms (135.07 ms) | 2.07 | 4.95 | 3.75 | 7.75 ms/7.30 ms | 128.0 KiB | 1.7 MiB/7.0 MiB | 0.0e+00 |
| chain(n=8,depth=4096) | 49166/49165 | 35.32 ms | 48.16 ms | 58.18 ms | 191.13 ms | 92.16 ms (104.17 ms) | 241.89 ms (274.25 ms) | 1.36 | 3.29 | 2.62 | 17.50 ms/14.12 ms | 256.0 KiB | 3.5 MiB/14.0 MiB | 0.0e+00 |
| diamond(n=8,depth=16,width=4) | 814/733 | 160.1 us | 350.1 us | 378.3 us | 1.32 ms | 544.3 us (735.3 us) | 1.68 ms (1.86 ms) | 2.19 | 3.48 | 3.10 | 138.3 us/150.5 us | 2.6 KiB | 33.8 KiB/174.1 KiB | 0.0e+00 |
| diamond(n=8,depth=64,width=4) | 3214/2893 | 1.29 ms | 1.66 ms | 2.56 ms | 5.96 ms | 3.87 ms (4.59 ms) | 7.69 ms (9.85 ms) | 1.28 | 2.33 | 1.99 | 896.1 us/1.39 ms | 10.1 KiB | 139.8 KiB/727.4 KiB | 0.0e+00 |
| diamond(n=8,depth=256,width=4) | 12814/11533 | 4.46 ms | 8.60 ms | 9.06 ms | 32.23 ms | 13.58 ms (14.60 ms) | 40.87 ms (45.12 ms) | 1.93 | 3.56 | 3.01 | 2.89 ms/2.91 ms | 40.1 KiB | 560.0 KiB/2.9 MiB | 0.0e+00 |
| diamond(n=8,depth=1024,width=4) | 51214/46093 | 22.45 ms | 42.77 ms | 42.26 ms | 153.23 ms | 65.60 ms (80.56 ms) | 199.00 ms (234.69 ms) | 1.90 | 3.63 | 3.03 | 16.75 ms/12.58 ms | 160.1 KiB | 2.2 MiB/11.6 MiB | 0.0e+00 |
| mlp(batch=4,in=8,hidden=8,out=8,layers=8) | 296/281 | 49.8 us | 90.2 us | 164.4 us | 379.1 us | 215.3 us (347.6 us) | 486.7 us (578.4 us) | 1.81 | 2.31 | 2.26 | 75.9 us/39.7 us | 2.9 KiB | 11.6 KiB/29.7 KiB | 0.0e+00 |
| mlp(batch=4,in=8,hidden=8,out=8,layers=32) | 1040/977 | 191.7 us | 297.6 us | 745.7 us | 1.35 ms | 958.7 us (1.16 ms) | 1.71 ms (1.85 ms) | 1.55 | 1.81 | 1.79 | 144.9 us/138.6 us | 9.7 KiB | 49.5 KiB/130.7 KiB | 0.0e+00 |
| mlp(batch=4,in=8,hidden=8,out=8,layers=128) | 4016/3761 | 768.6 us | 2.32 ms | 2.52 ms | 9.32 ms | 3.32 ms (3.62 ms) | 11.72 ms (12.64 ms) | 3.01 | 3.69 | 3.54 | 589.5 us/1.05 ms | 36.7 KiB | 202.3 KiB/555.0 KiB | 0.0e+00 |

## T6 One-time costs vs steady execution (DAG) and first execution

| configuration | build (cold) | diff (cold) | compile | first exec DAG | steady exec DAG | first exec torch | steady torch | liveness: peak live / all defined |
|---|---|---|---|---|---|---|---|---|
| chain(n=1024,depth=16) | 24.2 us (32.7 us) | 29.8 us (38.5 us) | 12.1 us | 438.3 us | 234.9 us | 785.3 us | 724.4 us | 132.0 KiB / 400.0 KiB |
| chain(n=65536,depth=256) | 396.1 us (465.4 us) | 444.3 us (479.4 us) | 153.0 us | 178.60 ms | 98.92 ms | 80.48 ms | 72.93 ms | 128.3 MiB / 400.0 MiB |
| chain(n=8,depth=4096) | 23.40 ms (17.96 ms) | 31.83 ms (29.07 ms) | 14.80 ms | 85.95 ms | 92.16 ms | 219.92 ms | 241.89 ms | 256.0 KiB / 800.0 KiB |
| diamond(n=8,depth=1024,width=4) | 11.19 ms (9.18 ms) | 19.88 ms (19.35 ms) | 10.06 ms | 47.80 ms | 65.60 ms | 192.58 ms | 199.00 ms | 160.1 KiB / 864.0 KiB |
| mlp(batch=128,in=1024,hidden=1024,out=64,layers=8) | 21.2 us (35.4 us) | 36.3 us (52.0 us) | 16.1 us | 46.09 ms | 31.73 ms | 27.77 ms | 28.10 ms | 33.8 MiB / 54.0 MiB |
| nn(batch=1024,in=1024,out=1024) | 2.4 us (13.2 us) | 6.2 us (13.2 us) | 4.8 us | 25.15 ms | 20.35 ms | 25.48 ms | 25.29 ms | 16.0 MiB / 25.0 MiB |
