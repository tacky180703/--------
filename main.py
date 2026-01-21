import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def generate_point_queue(image_path, epsilon_val=5.0):
    # 画像の読み込み
    if not os.path.exists(image_path):
        print(f"エラー: ファイル '{image_path}' が見つかりません。")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"エラー: 画像をデコードできませんでした。形式を確認してください。")
        return None

    # 二値化と細線化
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    try:
        skeleton = cv2.ximgproc.thinning(thresh)
    except AttributeError:
        print("エラー: cv2.ximgproc が見つかりません。'pip install opencv-contrib-python' を実行してください。")
        # ximgprocがない場合の代替手段（簡易的な細線化）
        kernel = np.ones((3,3), np.uint8)
        skeleton = cv2.erode(thresh, kernel, iterations=1)

    # パス抽出
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 座標格納
    point_queue = deque()
    
    # すでに登録したパスの「端点の集合」を記録するリスト
    processed_endpoints = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon_val, closed=False)
        if len(approx) < 2:
            continue

        # このパスの始点と終点を取得
        start_pt = (int(approx[0][0][0]), int(approx[0][0][1]))
        end_pt = (int(approx[-1][0][0]), int(approx[-1][0][1]))
        
        # --- 重複チェック ---
        is_duplicate = False
        for old_start, old_end in processed_endpoints:
            # 始点と始点、終点と終点が近い、あるいは「逆向き（始点と終点）」が近いか判定
            dist1 = np.linalg.norm(np.array(start_pt) - np.array(old_start))
            dist2 = np.linalg.norm(np.array(end_pt) - np.array(old_end))
            dist3 = np.linalg.norm(np.array(start_pt) - np.array(old_end))
            dist4 = np.linalg.norm(np.array(end_pt) - np.array(old_start))
            
            # 閾値（例: 10ピクセル）以内のズレなら同じ線とみなす
            threshold = 10.0
            if (dist1 < threshold and dist2 < threshold) or (dist3 < threshold and dist4 < threshold):
                is_duplicate = True
                break
        
        if is_duplicate:
            continue # すでに似た線があるのでスキップ

        # 重複していなければ追加
        processed_endpoints.append((start_pt, end_pt))
        for i in range(len(approx)):
            x, y = approx[i][0]
            point_queue.append((int(x), int(y)))
            
        point_queue.append(None)

    return point_queue, img

def visualize_results(point_queue, original_img):
    """
    生成されたキューの内容を可視化(消してもいい)
    """
    if not point_queue:
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(original_img, cmap='gray')
    
    q_list = list(point_queue)
    
    curr_x, curr_y = [], []
    point_count = 0

    for p in q_list:
        if p is None:
            plt.plot(curr_x, curr_y, marker='o', markersize=3, linewidth=2, label=f"Path Segment" if point_count==0 else "")
            curr_x, curr_y = [], []
        else:
            curr_x.append(p[0])
            curr_y.append(p[1])
            point_count += 1

    plt.title(f"Generated Path (Total Points: {point_count})")
    plt.xlabel("X [pixel]")
    plt.ylabel("Y [pixel]")
    plt.show()

if __name__ == "__main__":
    image_dir = "image"
    image_filename = sys.argv[1] if len(sys.argv) > 1 else 'test.png'
    target_path = os.path.join(image_dir, image_filename)
    
    if os.path.exists(target_path):
        # ルート（巡る点を順に格納したキュー)生成
        result = generate_point_queue(target_path, epsilon_val=8.0)
        
        if result:  
            # 出力（x,y)     
            q, original_img = result     
            temp_q = q.copy()
            while temp_q:
                p = temp_q.popleft()
                if p is not None:
                    print(f"{p}")
            
            # 表示（消してもいい）
            # visualize_results(q, original_img)