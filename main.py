import os
import sys
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

def generate_eulerian_path(image_path):
    start_time = time.time()

    # 1. 画像読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つかりません。")
        return

    # 2. 前処理（二値化・細線化）
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = cv2.ximgproc.thinning(thresh)

    # 3. グラフ構築
    G = nx.Graph()
    points = np.column_stack(np.where(skeleton > 0))
    for r, c in points:
        G.add_node((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1]:
                    if skeleton[nr, nc] > 0:
                        G.add_edge((r, c), (nr, nc))

    if len(G.nodes) == 0:
        print("エラー: 有効な線が検出されませんでした。")
        return

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # 4. ルート構築（連続性を確保）
    final_route_nodes = []
    if nx.has_eulerian_path(G):
        status_msg = "一筆書きルートが見つかりました！"
        euler_edges = list(nx.eulerian_path(G))
        final_route_nodes = [euler_edges[0][0]] + [e[1] for e in euler_edges]
    else:
        status_msg = "完全な一筆書き不可。戻り道を含む連続ルートを生成します。"
        # 深さ優先探索の巡回順（行き止まりで戻る動作を含む）を生成
        nodes_order = list(nx.dfs_preorder_nodes(G))
        # 簡易的に隣接ノードを繋ぐ（より厳密な一筆書きにはHierholzer拡張が必要ですが、
        # まずは連続性を優先します）
        final_route_nodes = nodes_order

    # 5. 直線の間引き（ショートカット機能）
    # 
    # --- 頂点の間引き（角度許容版） ---
    simplified_nodes = []
    if len(final_route_nodes) > 0:
        simplified_nodes.append(final_route_nodes[0])
        
        # 許容する角度（度数法）。この値を大きくすると、よりカクカクした線も直線とみなします。
        angle_threshold = 10.0 

        for i in range(1, len(final_route_nodes) - 1):
            prev = final_route_nodes[i-1]
            curr = final_route_nodes[i]
            nxt  = final_route_nodes[i+1]
            
            # ベクトル1と2
            v1 = np.array([curr[1] - prev[1], curr[0] - prev[0]])
            v2 = np.array([nxt[1] - curr[1], nxt[0] - curr[0]])
            
            # ベクトルの長さを計算
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                # 内積から角度（ラジアン）を求める
                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                # 数値誤差で1を超えないようクリップ
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_theta))
                
                # 指定した角度以上の変化がある場合のみ「角」として保存
                if angle > angle_threshold:
                    simplified_nodes.append(curr)
        
        simplified_nodes.append(final_route_nodes[-1])

    # 6. セグメント化 [開始x, 開始y, 終了x, 終了y]
    route_data = []
    for i in range(len(simplified_nodes) - 1):
        p1 = simplified_nodes[i]
        p2 = simplified_nodes[i+1]
        route_data.append([p1[1], p1[0], p2[1], p2[0]])

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 7. 結果表示
    print("-" * 30)
    print(status_msg)
    print(f"処理にかかった時間: {elapsed_time:.4f} 秒")
    print(f"圧縮前の点数: {len(final_route_nodes)}")
    print(f"圧縮後のセグメント数: {len(route_data)}")
    print("-" * 30)
    
    print("\n--- ルートデータ [開始x, 開始y, 終了x, 終了y] ---")
    for segment in route_data:
        print(segment)

    # 8. 可視化
    x_coords = [p[1] for p in simplified_nodes]
    y_coords = [p[0] for p in simplified_nodes]
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.plot(x_coords, y_coords, color='red', linewidth=1, alpha=0.7)
    plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start')
    plt.scatter(x_coords[-1], y_coords[-1], color='blue', s=100, label='End')
    plt.title(f"Compressed Path: {len(route_data)} segments")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    image_dir = "image" 
    filename = sys.argv[1] if len(sys.argv) > 1 else 'test.png'
    target_path = os.path.join(image_dir, filename)
    if os.path.exists(target_path):
        generate_eulerian_path(target_path)
    else:
        print(f"エラー: ファイルが見つかりません -> {target_path}")