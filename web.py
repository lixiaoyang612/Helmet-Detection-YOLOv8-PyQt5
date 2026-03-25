
import os
import queue
import time
import random
import csv
import traceback
import cv2
import zipfile
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from threading import Thread, Event, Lock
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'pt', 'onnx', 'torchscript', 'engine', 'mlmodel', 'pb', 'tflite', 'jpg', 'jpeg', 'png', 'bmp', 'dng', 'mp4', 'avi', 'mov', 'zip'}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# 全局变量
current_session = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
session_output_path = os.path.join(app.config['OUTPUT_FOLDER'], current_session)
result_img_path = os.path.join(session_output_path, 'img_result')
result_org_img_path = os.path.join(session_output_path, 'org_img')
os.makedirs(result_img_path, exist_ok=True)
os.makedirs(result_org_img_path, exist_ok=True)

# 初始化模型和状态
model = None
weights_file_name = ""
weights_file_old_name = ""
names = None
color = {"font": (255, 255, 255)}
inference_thread = None
stop_event = Event()
results_data = []
current_session_files = {}
active_inference = False
inference_running = False

# 使用线程安全的队列
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue()
frame_counter = 0
shared_frame = None
shared_frame_lock = Lock()  # 用于保护共享帧的锁


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def init_model():
    global model, names, color, weights_file_name
    if weights_file_name and weights_file_name != weights_file_old_name:
        model = YOLO(model=weights_file_name)
        names = model.names
        color.update({names[i]: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in
                      range(len(names))})


def inference_worker(conf, iou, classes, stop_signal):
    global model, names, frame_queue, result_queue, shared_frame, shared_frame_lock, frame_counter

    processed_count = 0
    while not stop_signal.is_set():
        try:
            # 从队列获取帧，超时100ms
            data = frame_queue.get(timeout=0.1)
            frame_idx, frame = data
            processed_count += 1

            # 执行推理
            results = model(frame, conf=conf, iou=iou, classes=classes, stream=True)

            all_result = []
            result_img = frame.copy()  # 保存原始帧用于绘制

            for result in results:
                # 处理每个检测结果
                for boxs, cls, conf_val in zip(result.boxes.xyxy.tolist(),
                                               result.boxes.cls.tolist(),
                                               result.boxes.conf.tolist()):
                    all_result.append([round(i, 2) for i in boxs] +
                                      [names[int(cls)]] +
                                      [round(conf_val, 4)])

                # 绘制结果
                result_img = draw_info(frame.copy(), all_result, color)

            # 将结果放入结果队列
            result_queue.put((frame_idx, result_img, all_result))

            # 更新共享帧
            with shared_frame_lock:
                _, jpeg = cv2.imencode('.jpg', result_img)
                shared_frame = jpeg.tobytes()
                frame_counter = frame_idx

        except queue.Empty:
            # 队列为空时继续循环
            continue
        except Exception as e:
            print(f"推理错误: {e}")
            # 发生错误时放入空结果防止阻塞
            traceback.print_exc()
            result_queue.put((frame_idx, frame.copy(), []))
            continue


def draw_info(draw_img, results, color_dict):
    lw = max(round(sum(draw_img.shape) / 2 * 0.003), 2)  # line width
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  # font scale

    for idx, result in enumerate(results):
        box, name, conf = result[:4], result[4], result[5]
        color_val = color_dict.get(name, (0, 0, 255))  # 默认红色
        label = f'{name} {conf}'

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(draw_img, p1, p2, color_val, thickness=lw, lineType=cv2.LINE_AA)

        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        outside = box[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(draw_img, p1, p2, color_val, -1, cv2.LINE_AA)

        cv2.putText(draw_img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, sf, color_dict["font"], thickness=2, lineType=cv2.LINE_AA)
    return draw_img


def inference_task(start_type, file_path=None, cap_source=None, conf=None, iou=None, classes=None):
    global stop_event, current_session_files, inference_running, active_inference, inference_worker_thread, frame_queue, result_queue, shared_frame
    inference_running = True
    active_inference = True
    stop_event.clear()
    stop_signal = Event()

    # 清空队列
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except:
            pass

    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except:
            pass

    # 启动推理工作线程
    inference_worker_thread = Thread(
        target=inference_worker,
        args=(conf, iou, classes, stop_signal)
    )
    inference_worker_thread.daemon = True
    inference_worker_thread.start()

    if start_type == 'img':
        start_time = time.time()

        img = cv2.imread(file_path)
        img_name = os.path.basename(file_path)

        # 首先保存原始图片
        org_img_save_path = os.path.join(result_org_img_path, img_name)
        cv2.imwrite(org_img_save_path, img)

        frame_queue.put((0, img))

        while True:
            if not result_queue.empty():
                idx, result_frame, all_result = result_queue.get()

                result_img_name = os.path.join(result_img_path, f"{img_name}_{idx}.jpg")
                cv2.imwrite(result_img_name, result_frame)
                end_time = str(round(time.time() - start_time, 4)) + 's'

                results_data.append({
                    'org_img_path': org_img_save_path,
                    'input_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'all_result': all_result,
                    'num_targets': len(all_result),
                    'infer_time': end_time,
                    'result_img_name': result_img_name,
                    'results_index': {f'目标{i + 1}': res for i, res in enumerate(all_result)}
                })
                break
            time.sleep(0.05)

    elif start_type == 'dir':
        frame_idx = 0
        if file_path in current_session_files:
            image_files = current_session_files[file_path]
            for img_file in image_files:
                if stop_event.is_set():
                    break

                start_time = time.time()
                img = cv2.imread(img_file)
                img_name = os.path.basename(img_file)

                org_img_save_path = os.path.join(result_org_img_path, img_name)
                cv2.imwrite(org_img_save_path, img)

                frame_queue.put((frame_idx, img))
                frame_idx += 1

                # 等待结果
                while True:
                    if not result_queue.empty():
                        idx, result_frame, all_result = result_queue.get()
                        result_img_name = os.path.join(result_img_path, img_name)
                        cv2.imwrite(result_img_name, result_frame)
                        end_time = str(round(time.time() - start_time, 4)) + 's'

                        results_data.append({
                            'org_img_path': org_img_save_path,
                            'input_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'all_result': all_result,
                            'num_targets': len(all_result),
                            'infer_time': end_time,
                            'result_img_name': result_img_name,
                            'results_index': {f'目标{i + 1}': res for i, res in enumerate(all_result)}
                        })
                        break
                    time.sleep(0.01)

    elif start_type in ['video', 'cap']:
        if start_type == 'video':
            video_path = file_path
            video_name = os.path.basename(video_path)[:4]
        else:
            video_path = int(cap_source) if cap_source.isdigit() else cap_source
            video_name = 'camera'

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {'status': 'error', 'message': '无法打开视频文件'}

        fps = cap.get(cv2.CAP_PROP_FPS) if video_name != 'camera' else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入器
        save_path = os.path.join(result_img_path, video_name + ".mp4")
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_idx = 0
        while not stop_event.is_set():
            start_time = time.time()

            ret, frame = cap.read()

            org_img_save_path = os.path.join(result_org_img_path, f'{video_name}_{frame_idx}.jpg')
            cv2.imwrite(org_img_save_path, frame)

            if not ret:
                break

            # 放入队列（如果队列满则跳过当前帧）
            if frame_queue.qsize() < 20:
                frame_queue.put((frame_idx, frame))
                frame_idx += 1
            else:
                # 队列满时跳过帧，避免阻塞
                continue

            # 从结果队列获取结果
            while not result_queue.empty():
                r_idx, r_frame, r_result = result_queue.get()
                vid_writer.write(r_frame)

                temp_path = os.path.join(result_img_path, f"{video_name}_{r_idx}.jpg")
                cv2.imwrite(temp_path, r_frame)
                end_time = str(round(time.time() - start_time, 4)) + 's'

                results_data.append({
                    'org_img_path': org_img_save_path,
                    'input_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'all_result': r_result,
                    'num_targets': len(r_result),
                    'infer_time': end_time,
                    'result_img_name': temp_path,
                    'results_index': {f'目标{i + 1}': res for i, res in enumerate(r_result)}
                })

            # 控制处理速率
            time.sleep(max(0, 1 / fps - 0.01))

        cap.release()
        vid_writer.release()

    # 停止推理工作线程
    stop_signal.set()
    inference_worker_thread.join(timeout=1.0)
    inference_running = False
    active_inference = False
    return {'status': 'completed'}


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_model', methods=['POST'])
def upload_model():
    global weights_file_name
    if 'model' not in request.files:
        return jsonify({'error': '未选择文件'}), 400

    file = request.files['model']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['MODEL_FOLDER'], filename)
        file.save(save_path)
        weights_file_name = save_path
        init_model()
        return jsonify({'success': True, 'filename': filename})

    return jsonify({'error': '文件类型不允许'}), 400


@app.route('/upload_file', methods=['POST'])
def upload_file():
    global current_session_files
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # 处理ZIP文件（文件夹上传）
        if filename.lower().endswith('.zip'):
            try:
                extract_to = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename[:-4]}_{uuid.uuid4().hex}")
                os.makedirs(extract_to, exist_ok=True)

                # 确保ZIP文件存在
                if not os.path.exists(save_path):
                    return jsonify({'error': 'ZIP文件不存在'}), 400

                # 解压ZIP文件
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)

                # 获取所有图片文件
                image_files = []
                for root, _, files in os.walk(extract_to):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            # 使用绝对路径
                            image_files.append(os.path.abspath(os.path.join(root, f)))

                if not image_files:
                    return jsonify({'error': 'ZIP文件中未找到图片文件'}), 400

                current_session_files[save_path.replace("\\", "/")] = image_files
                return jsonify({
                    'success': True,
                    'filepath': save_path,
                    'filename': f"{os.path.basename(save_path)}",
                    'file_count': len(image_files)
                })
            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': f'解压ZIP文件失败: {str(e)}'}), 500
        else:
            # 单个文件
            return jsonify({
                'success': True,
                'filepath': save_path,
                'filename': filename,
                'file_count': 1
            })

    return jsonify({'error': '文件类型不允许'}), 400


@app.route('/start_inference', methods=['POST'])
def start_inference():
    global inference_thread

    start_type = request.form.get('start_type')
    file_path = request.form.get('file_path', '')
    cap_source = request.form.get('cap_source', '0')

    if not model:
        return jsonify({'error': '请先上传模型文件'}), 400

    # 停止正在运行的推理线程
    if inference_thread and inference_thread.is_alive():
        stop_event.set()
        inference_thread.join()

    # 启动新的推理线程
    inference_thread = Thread(target=inference_task, args=(start_type, file_path, cap_source,
                                                           float(request.form.get('conf', 0.45)),
                                                           float(request.form.get('iou', 0.25)),
                                                           [int(i) for i in request.form.get('classes', '').split(
                                                               ",")] if request.form.get('classes', '') != '' else None))
    inference_thread.start()
    return jsonify({'success': True})


@app.route('/stop_inference', methods=['POST'])
def stop_inference():
    global inference_thread, inference_worker_thread

    # 停止推理线程
    if inference_thread and inference_thread.is_alive():
        stop_event.set()
        inference_thread.join()

    # 停止推理工作线程
    if inference_worker_thread and inference_worker_thread.is_alive():
        # 设置停止信号并等待线程结束
        inference_worker_thread.join(timeout=1.0)

    # 清空队列
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except:
            break

    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except:
            break

    return jsonify({'success': True})


@app.route('/get_results')
def get_results():
    return jsonify({
        'results': results_data,
        'current_index': len(results_data) - 1 if results_data else -1,
        'active_inference': active_inference
    })


@app.route('/get_image')
def get_image():
    index = request.args.get('index', type=int)
    target_index = request.args.get('target_index', type=int)

    if index is None or index < 0 or index >= len(results_data):
        return jsonify({'error': '无效的索引'}), 400

    result_data = results_data[index]
    # 使用原始图片而不是推理后的图片
    orig_img_path = result_data['org_img_path']

    if not os.path.exists(orig_img_path):
        return jsonify({'error': '原始图片不存在'}), 404

    # 读取原始图像
    orig_img = cv2.imread(orig_img_path)

    # 绘制指定目标
    if target_index is not None and target_index >= 0:
        # 只绘制选定的目标
        target_data = [result_data['all_result'][target_index]]
        draw_img = draw_info(orig_img.copy(), target_data, color)
    else:
        # 绘制所有目标
        draw_img = draw_info(orig_img.copy(), result_data['all_result'], color)

    # 保存临时图像
    temp_path = os.path.join(result_img_path, f"temp_{os.path.basename(orig_img_path)}")
    cv2.imwrite(temp_path, draw_img)
    return send_file(temp_path)


@app.route('/get_latest_frame')
def get_latest_frame():
    global shared_frame, shared_frame_lock, inference_running

    # 只有推理运行时才返回最新帧
    if inference_running:
        with shared_frame_lock:
            if shared_frame:
                return Response(shared_frame, mimetype='image/jpeg')
    return Response(b'', mimetype='image/jpeg')


@app.route('/export_csv')
def export_csv():
    csv_path = os.path.join(session_output_path, 'result.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['序号', '图片名称', '录入时间', '识别结果', '目标数目', '推理用时', '保存路径'])

        for i, result in enumerate(results_data):
            writer.writerow([
                i + 1,
                os.path.basename(result['org_img_path']),
                result['input_time'],
                str(result['all_result']),
                result['num_targets'],
                result['infer_time'],
                result['result_img_name']
            ])
    # return send_file(csv_path, as_attachment=True)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=False, port=5000)