import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_eulerian_path(image_path):
    # 1. 画像の読み込みと二値化
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 背景を黒、線を白にする（適宜反転させてください）
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 2. 細線化 (Skeletonization)
    # 線の幅を1ピクセルにする
    skeleton = cv2.ximgproc.thinning(thresh)

    # 3. グラフの構築
    # 白いピクセルをノードとして、隣接するピクセル間にエッジを張る
    G = nx.Graph()
    points = np.column_stack(np.where(skeleton > 0))
    
    for r, c in points:
        G.add_node((r, c))
        # 周囲8近傍をチェック
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1]:
                    if skeleton[nr, nc] > 0:
                        G.add_edge((r, c), (nr, nc))

    # 4. 一筆書き（オイラー路）の計算
    # グラフが連結でない場合は、最大の連結成分のみを対象にする
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # オイラー路が存在するか確認し、必要なら補完（簡易版のため判定のみ）
    if nx.has_eulerian_path(G):
        path = list(nx.eulerian_path(G))
        print("一筆書きルートが見つかりました！")
    else:
        # 一筆書きできない場合、最短経路でエッジを足して「オイラー化」する処理が必要
        # ここでは近似的なルートとして「深さ優先探索(DFS)」で代用
        print("完全な一筆書きができない形状のため、DFSでルートを生成します。")
        path = list(nx.dfs_edges(G))

    # 5. 結果の可視化
    y_coords = [p[0][0] for p in path]
    x_coords = [p[0][1] for p in path]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.plot(x_coords, y_coords, color='red', linewidth=2, label='Path')
    plt.scatter(x_coords[0], y_coords[0], color='green', label='Start') # 開始点
    plt.scatter(x_coords[-1], y_coords[-1], color='blue', label='End')  # 終了点
    plt.legend()
    plt.title("Generated One-Stroke Path")
    plt.show()

# 実行
# generate_eulerian_path('your_image.png')