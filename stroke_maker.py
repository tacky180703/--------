import sys
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time  # 時間計測用のライブラリを追加

def generate_eulerian_path(image_path):
    # 計測開始
    start_time = time.time()

    # 1. 画像の読み込みと二値化
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つかりません。")
        return

    # 背景を黒、線を白にする（適宜反転させてください）
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 2. 細線化 (Skeletonization)
    skeleton = cv2.ximgproc.thinning(thresh)

    # 3. グラフの構築
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
    if len(G.nodes) == 0:
        print("エラー: 有効な線が検出されませんでした。")
        return

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    if nx.has_eulerian_path(G):
        path = list(nx.eulerian_path(G))
        status_msg = "一筆書きルートが見つかりました！"
    else:
        status_msg = "完全な一筆書きができない形状のため、DFSでルートを生成します。"
        path = list(nx.dfs_edges(G))

    # 計測終了
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 結果の報告
    print("-" * 30)
    print(status_msg)
    print(f"処理にかかった時間: {elapsed_time:.4f} 秒")
    print("-" * 30)

    # 5. 結果の可視化
    y_coords = [p[0][0] for p in path]
    x_coords = [p[0][1] for p in path]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.plot(x_coords, y_coords, color='red', linewidth=2, label='Path')
    plt.scatter(x_coords[0], y_coords[0], color='green', label='Start') # 開始点
    plt.scatter(x_coords[-1], y_coords[-1], color='blue', label='End')  # 終了点
    plt.legend()
    plt.title(f"Generated Path (Time: {elapsed_time:.2f}s)")
    plt.show()

# 実行
if __name__ == "__main__":
    # コマンドライン引数があればそれを使う、なければ test.png を使う
    target_file = sys.argv[1] if len(sys.argv) > 1 else 'test.png'
    generate_eulerian_path(target_file)