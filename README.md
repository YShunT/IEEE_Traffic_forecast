# 2026 IEEE Big Data-Traffic Flow Bench
[2026 IEEE Big Data Cup](https://bigdataieee.org/BigData2026/cup/) の高速道路交通ベンチマーク（Kaggle）に取り組むリポジトリ。10の方向別コリドーの5分間隔データ（合成）から、交通状態の再構築・渋滞予測・物理的整合性・OD推定の4タスクを解く。

## 結果

| Public | Private | 順位 |
| ---: | ---: | :--- |
| 0.87309 | 0.87105 | 111 / 801 |


## タスク
合計スコアは4タスクの加重和。各タスクは10パネルで計算し、コリドーの2方向を平均してから5コリドーを平均する。

$$
S_{\mathrm{total}} = 0.35\,S_{\mathrm{state}} + 0.30\,S_{\mathrm{queue}} + 0.15\,S_{\mathrm{physics}} + 0.20\,S_{\mathrm{ODME}}
$$

| タスク | 重み | 内容 | ローカルの CV |
| :--- | ---: | :--- | :--- |
| 1 状態再構築 | 0.35 | 空欄にされたセルの速度・交通量を復元（速度 RMSE と車線あたり交通量 RMSE） | `S_state`：train の最後の1か月（2031-02）で検証 |
| 2 渋滞予測 | 0.30 | 60分の履歴から30分先の渋滞（0/1）を予測（時空間 IoU） | EXP07：2月の擬似窓・代理教師によるIoU（ongoing 390窓、onset 235窓）。参考値として使用 |
| 3 物理整合性 | 0.15 | Task1 の値が基本図と車両保存則に従うか（提出行なし） | `S_FD` と、ローカルの近似の `S_LWR`（境界流量が非公開のため。EXP03 から） |
| 4 OD推定 | 0.20 | リンクカウントと弱い事前値から経路交通量を推定 | `S_link` のみ（ほかは真の経路交通量が非公開） |

## 公開ベースラインのスコア内訳

配布元の [BASELINES.md](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/BASELINES.md) に掲載された validation の参考値。寄与は「重み × スコア」で計算した。EXP00 の Kaggle 提出ではタスク別スコアは公開されない。

| タスク | 重み | ベースラインのスコア | 合計への寄与 |
| :--- | ---: | ---: | ---: |
| 1 状態再構築 | 0.35 | 0.6929 | 0.2425 |
| 2 渋滞予測 | 0.30 | 0.3017 | 0.0905 |
| 3 物理整合性 | 0.15 | 0.3549 | 0.0532 |
| 4 OD推定 | 0.20 | 0.8359 | 0.1672 |
| **合計** | **1.00** | — | **0.5534** |

パブリック LB は validation（2031-03）、プライベート LB は private（2031-04）で採点される。

## 手法

### タスク 1・3
時間×道路の 2 次元 U-Net を全パネル共通で 1 つ学習し、空欄の速度・交通量を復元する。密度も出力させて答えを密度と整合する方へ寄せ、Task 3（物理整合性）の点数も上げる。

### タスク 2
渋滞がこれから始まる区間（onset）とすでに続いている区間（ongoing）を別々に予測する。どちらも LightGBM と、道路グラフ上の時系列モデル（DCRNN 型）などの確率を平均し、0.5 で切って渋滞の有無を決める。

### タスク 4
区間交通量と事前値から、経路交通量を事前値つきの非負最小二乗で解く。

## コード構成

```text
src/
├── main.py            入口（工程の一覧）
├── run.py             全工程の実行（CPU と GPU の工程を並行）
├── submit.py          提出ファイルの組み立て
├── config.json        設定
├── core/              設定・定数・出力先（config.py）、デバイス・GPU・ログ（runtime.py）
├── data/              配布データの読み込み（release.py）、パネルの配列（panel.py）
├── task1_3/           U-Net：材料（data）→ GPU 上の配列（store）→ モデル（model）→ 学習（train）→ 工程（part）
├── task2/
│   ├── dcrnn.py       共通の DCRNN 型モデル
│   ├── onset_place/   共通（data）、GBM（gbm）、DCRNN の入力（dcrnn_data）と工程（dcrnn）
│   ├── onset_signs/   共通（features）、GBM（gbm）、DCRNN（dcrnn）
│   ├── onset_mix.py   onset の混合
│   └── ongoing/       共通（common）、GBM・DCRNN・U-Net、平均（ensemble）
└── task4.py           非負最小二乗
```
