## 更新履歴
- 2021/4/14 v1.0: 課題公開
- 2021/4/23 v1.1: `src/lr_sample.py` の更新
  - train-val 分割をアミノ酸配列のレベルで行うようにした

# 研究室対抗型新人課題

新人課題の最後は、研究室対抗型の課題になっています。

研究を進める上で、様々な大学の授業を受ける上で、あるいは就職活動を行う上で、研究室の同期や先輩（や後輩）とコミュニケーションをとることは大切です。
一方でコロナ禍では自分から「よし、しゃべるために準備しよう！」としないとコミュニケーションをとることができず、やり取りが希薄になりがちです。

この課題は、お互いにはじめましての状態から、まずは研究室内の仲間同士でコミュニケーションをとる（取らざるを得ない）材料として設定しています。
課題の作業に関する相談の前後にこの課題とは関係ない話をする時間を設けるなど、各々の工夫で仲間を増やしていきましょう。

## 課題背景：タンパク質二次構造

タンパク質の二次構造とは、タンパク質主鎖のアミノ酸残基のアミノ基 (N-H) とカルボニル基 (C=O) のあいだに結ばれる水素結合 (N-H…O=C) による形成される局所的な規則的構造のことで、代表的なものにαヘリックス、βシートがあります。

この課題では特にβシートを取り上げます。βシートはポリペプチドが伸びた構造をしており、隣り合ったペプチド鎖間で水素結合を形成し、シート（板）状の構造をとります。

![2ar9](https://user-images.githubusercontent.com/6902135/113391892-36b0c580-93cf-11eb-93e3-9f318f382c75.png)
**図1 | タンパク質立体構造の例 (Caspase-9, PDB ID: 2AR9)** 太い矢印で表現されている部分がβシート構造です。

## 課題内容

この課題では、**タンパク質のアミノ酸配列のうち、どの残基がβシートを形成するかを予測**してください。

ただし、グループワークをする上で、以下の条件を設けます。
- **グループの各人が独立のβシート予測モデルを構築する**こと
  - 予測モデルは各予測対象残基に対して0～1の予測値を出力する
  - **グループ内で予測方法が重複しないように各自で調整する**こと
    - NGな例：「SVM（RBFカーネル）」と「SVM（Linearカーネル）」は予測方法が重複しているとみなす
    - OKな例：「SVM」と「PCAを行ってからSVM」は異なる予測方法と扱う
- グループの各人の予測モデルが出力した予測値を統合し、各予測対象残基に対して0～1の予測値を出力、そのグループの予測値とする
  - 統合の方法：単純平均、加重平均、予測器のスタッキングなど
  - **最終提出物はグループの予測値としてください**。

## データセット
`data/train.dat` に訓練データが、`data/test.dat` にテストデータが配置されています。

### 訓練データ
訓練データは以下の3列がカンマ区切りフォーマットで構成されており、1行が1つのアミノ酸配列に対応します。
  - sequence_id: アミノ酸配列のID。1から始まる。
  - sequence: アミノ酸配列。
  - label: アミノ酸配列に対応した、βシート構造を取っているか否かを示す0/1値。1の場所がβシート構造を取っている。
    - label は Stride を使って、Protein Data Bank (PDB) の立体構造から生成されています。

### テストデータ
テストデータは以下の2列がカンマ区切りフォーマットで構成されており、1行が1つのアミノ酸配列に対応します。
  - sequence_id: アミノ酸配列のID。訓練データとは重複しないIDになっている。
  - sequence: アミノ酸配列。

## 提出物の形式と評価

### 最終提出物の形式

提出するデータの形式は以下の列**のみ**が存在するカンマ区切りフォーマットにしてください。
  - sequence_id: テストデータのアミノ酸配列のID。
  - residue_number: アミノ酸配列の何残基目かを示す値。
  - predicted_value: 予測値（0以上1以下）

### 評価指標
評価指標は**ROC曲線の曲線下面積　(ROC-AUC)** を用います。

## 結果報告会
生命輪講1回分を使って、この新人課題についての報告会を行います（例年5月中旬頃を予定）
- 1研究室（グループ）あたり20分の発表時間、質疑は時間制限なし
  - 各人の構築したモデルの説明、モデルの統合方法の説明、テストデータ予測精度の説明
  - 各人の構築したモデルの説明は、その担当者本人が説明＆質疑応答すること

## スケジュール（日本時間; in JST）

- 4月14日（水） 課題公開
- 5月07日（金） グループのコード＆予測値を提出（to 柳澤）
  - **以下の場合は5月09日（日）までに再提出**
    - **一部もしくは全ての予測値の付与がない場合、およびレコードが不足している場合**
    - **予測値が0以上1以下に収まっていない場合**
- 5月08日（土）～5月09日（日） 採点
- 5月10日（月） テストデータのラベル**と各グループのROC-AUC値を公開**
- 5月17日（月）～5月21日（金） 結果報告会
  - 報告会は生命輪講を1回つぶして実施

## その他

- PDBにアクセスして、予測対象タンパク質の立体構造をカンニングすることは禁止です。
- 予測の例となるコードを `src/lr_predict.py` に配置しました。
  - `python src/lr_sample.py -train data/train.csv -test data/test.csv -out output.csv` で適切な形式に従った予測値ファイルを出力します。
  - 決して精度が良くないモデルなので、少なくともこれは上回るように頑張りましょう。
    - これに負ける場合はバグを疑った方がいいかも。

- 説明に不明瞭な点がある場合は柳澤まで問い合わせてください。
  - 記述を更新するなど、なるべく明確になるようにします（その場合には全体に伝えます）。


## 参考文献
『バイオインフォマティクス入門』（日本バイオインフォマティクス学会編、慶応義塾大学出版会、2015）第1-9節
