import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def generate_point_queue(image_path, epsilon_val=5.0):
    if not os.path.exists(image_path):
        print(f"エラー: ファイル '{image_path}' が見つかりません。")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"エラー: 画像をデコードできませんでした。")
        return None

    # 二値化と細線化
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    try:
        skeleton = cv2.ximgproc.thinning(thresh)
    except AttributeError:
        kernel = np.ones((3,3), np.uint8)
        skeleton = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    point_queue = deque()
    processed_endpoints = []
    tolerance = 10.0  # 往復判定の許容誤差（ピクセル）
    threshold = 10.0 # 重複パス判定の閾値

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon_val, closed=False)
        if len(approx) < 2:
            continue

        # 現在のパスをリスト化
        this_path = [(int(p[0][0]), int(p[0][1])) for p in approx]
        
        # 往復判定
        num_pts = len(this_path)
        mid_idx = int(num_pts/ 2)
        is_perfect_return = True
        
        for i in range(1, mid_idx + 1):
            idx_back = mid_idx - i
            idx_forw = mid_idx + i
            if idx_back < 0 or idx_forw >= num_pts:
                break
            
            p_back = np.array(this_path[idx_back])
            p_forw = np.array(this_path[idx_forw])
            
            if np.linalg.norm(p_back - p_forw) > tolerance:
                is_perfect_return = False
                break

        # 往復パスなら半分（片道）にする。そうでなければ（円など）そのまま。
        final_path = this_path[:mid_idx+1] if is_perfect_return else this_path

        # 重複判定
        start_pt = final_path[0]
        end_pt = final_path[-1]
        is_duplicate = False
        for old_start, old_end in processed_endpoints:
            dist1 = np.linalg.norm(np.array(start_pt) - np.array(old_start))
            dist2 = np.linalg.norm(np.array(end_pt) - np.array(old_end))
            dist3 = np.linalg.norm(np.array(start_pt) - np.array(old_end))
            dist4 = np.linalg.norm(np.array(end_pt) - np.array(old_start))
            
            if (dist1 < threshold and dist2 < threshold) or (dist3 < threshold and dist4 < threshold):
                is_duplicate = True
                break
        
        if is_duplicate:
            continue

        # 結果を登録
        processed_endpoints.append((start_pt, end_pt))
        point_queue.extend(final_path)
        point_queue.append(None) # パスの区切り

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
            visualize_results(q, original_img)