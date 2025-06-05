import torch

# requires_grad=True にしておくと、このテンソルを使った計算でグラフを作る
x = torch.randn(3, requires_grad=True)

# -- グラフを作るモード（no_grad なし） --------------------------------
y = x * 2          # 中間 y が計算グラフに登録される
z = y.mean()       # z も計算グラフに登録される
z.backward()       # ここで x.grad に勾配が計算される
print(x.grad)      # 勾配が取得できる

# -- メモリに残っている計算グラフを確認してみる -------------------------
print(y.grad_fn)   # ここをチェックすると、y を作る演算が残っているのがわかる
