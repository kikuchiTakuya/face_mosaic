# 外部ライブラリ
import re
import cv2

# 自作ライブラリ
from original_modules import cascade

# 顔に画像を乗せる処理
def img_on_face(w,h):
    # 画像読み込み
    img = cv2.imread("img/mark_face_hehe.png")
    # 画像を顔のサイズにリサイズ
    img = cv2.resize(img, (h, w))
    return img

# 顔にモザイクをかける処理
def face_mosaic(w,h,face):
    # 縮小する
    scale = 0.05 # 0 < scale <= 1
    ms = cv2.resize(face, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # 元のサイズに戻す
    ms = cv2.resize(ms, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    return ms


# 顔の部分を加工する処理
def face_proc(frame, faces_xywh, mode):
    for x, y, w, h in faces_xywh:
        face_frame = frame[y:y+h, x:x+w]
        if mode == 1:
            # 画像を乗せる
            face_frame = img_on_face(w,h)
        elif mode == 2:
            # モザイクをかける
            face_frame = face_mosaic(w,h,face_frame)
        else:
            pass
        frame[y:y+h, x:x+w] = face_frame

    return frame


def main():
    # カメラを使用
    CAP = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # 加工モード管理フラグ
    mode = 1

    while(True):
        # カメラから画像を取得
        r, frame = CAP.read()
        # 顔を検出
        # 顔のx,y,w,hを取得
        faces_xywh = cascade.judge_face(frame)
        # 顔の部分を加工
        show_frame = face_proc(frame, faces_xywh, mode)
        # 画像を表示
        cv2.imshow("output", show_frame)
        input_key = cv2.waitKey(1) & 0xFF
        # モードの変更
        if input_key == ord('i'):
            mode = 1
        elif input_key == ord('m'):
            mode = 2
        elif input_key == ord('n'):
            mode = 3

        if input_key == ord('q'):
            break
    
    # カメラを解放
    CAP.release()


if __name__ == "__main__":
    main()


