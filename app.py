import os
import subprocess
from flask import Flask, request, send_file
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
BIN_PATH = './ezSIFT/platforms/desktop/build/bin/plane_projection'

@app.route('/upload', methods=['POST'])
def upload_files():
    # 1. 接收兩張圖片
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    # 2. 轉檔為 PGM (因為 ezSIFT 只吃 PGM)
    path1 = os.path.join(UPLOAD_FOLDER, 'input1.pgm')
    path2 = os.path.join(UPLOAD_FOLDER, 'input2.pgm')
    Image.open(file1).convert('L').save(path1)
    Image.open(file2).convert('L').save(path2)

    # 3. 呼叫 C++ 執行檔 (這是關鍵！)
    # 假設 C++ 程式會讀入這兩個路徑，並將結果輸出到 result.pgm
    subprocess.run([BIN_PATH, path1, path2])

    # 4. 將結果轉回 JPG 並回傳
    result_pgm = 'result.pgm' # 假設 C++ 固定輸出這個檔名
    result_jpg = os.path.join(RESULT_FOLDER, 'output.jpg')
    Image.open(result_pgm).save(result_jpg)

    return send_file(result_jpg, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)