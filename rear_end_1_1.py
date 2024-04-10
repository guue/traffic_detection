import base64
import glob
import io
import json
import os
import socket
import threading
import traceback

import cv2
import numpy as np
from detect_v9 import VideoTracker, parse_opt
from connect import send_json_msg
from PIL import Image
# 假设connect.py包含了修改后的send_msg和send_msg1函数，以支持JSON格式的消息发送

class BackendServer:
    def __init__(self):
        self.status = "就绪"  # 初始状态为就绪
        self.video_tracker = None
        self.path = None
        self.client_sockets = []

    def update_status(self, new_status):
        """
        更新服务器状态并通知所有连接的客户端。
        :param new_status: 新的状态字符串
        """
        self.status = new_status
        # 构造状态更新消息
        status_message = {
            "action": "update_status",
            "status": self.status
        }
        # 向所有客户端发送状态更新消息
        for client_socket in self.client_sockets:
            self.send_json_msg(client_socket, status_message)

    def handle_connection(self, client_socket, address):
        self.client_sockets.append(client_socket)
        try:
            while True:
                # 接收JSON格式的消息
                message = self.recv_json_msg(client_socket)
                if message['action'] == 'set_video_source':
                    self.path = message['path']
                    self.update_status("ready")
                elif message['action'] == 'get_status':
                    self.update_status(self.status)
                elif message['action'] == 'set_status':
                    if message['data'] == 'start':
                        # 实现开始识别逻辑
                        self.update_status("recognizing")
                        opt = parse_opt()
                        self.video_tracker = VideoTracker(opt)
                        self.video_tracker.change_video(self.path)
                        self.video_tracker.start_detection()
                        self.update_status("stop")
                elif message['action'] == 'change_status' and message['data'] == 'stop' :
                            self.video_tracker.stop_detection()
                            self.update_status("stop")
                elif message['action'] == 'get_all_violations':
                            send_json_msg({"action":"return_all_violations",
                                           "data":self.video_tracker.get_illegal_car()})
                elif message['action'] == 'get_violation_image':  
                     id = message['data']
                     path = f"output/track/speedover/{id}/*.jpg"
                     img_path = glob.glob(path)[0]
                     image = Image.open(img_path)
                     buffered = io.BytesIO()
                     image.save(buffered, format="JPEG")
                     img_str = base64.b64encode(buffered.getvalue()).decode()
                     send_json_msg({"action":"return_violation_image",
                                    "data":{"id":int(id),
                                            "image_data":img_str}})
                     

        except Exception as e:
            # print(f"An error occurred: {str(e)}")
            traceback.print_exc()
        finally:
            self.client_sockets.remove(client_socket)
            client_socket.close()

    def recv_json_msg(self, client_socket):
        """
        接收客户端发送的JSON格式消息。
        :param client_socket: 客户端socket对象
        :return: 解析后的JSON数据（字典）
        """
        buffer_size = 1024
        data_str = ""
        while True:
            try:
                # 接收数据
                part = client_socket.recv(buffer_size).decode('utf-8')
                data_str += part
                if len(part) < buffer_size:
                    # 如果接收到的数据小于buffer_size，则认为数据已全部接收
                    break
            except socket.error as e:
                traceback.print_exc()
                # 处理接收数据时的错误
                # print(f"Socket error: {e}")
                break
        try:
            # 尝试将接收到的字符串解析为JSON
            data = json.loads(data_str)
            return data
        except json.JSONDecodeError as e:
            # 如果解析JSON失败，抛出异常
            traceback.print_exc()
            # print(f"JSON decode error: {e}")
            return None

    def send_json_msg(self, client_socket, message):
        """
        向客户端发送JSON格式的消息。
        :param client_socket: 客户端socket对象
        :param message: 要发送的消息（字典）
        """
        try:
            # 将消息字典转换为JSON字符串
            message_str = json.dumps(message)
            # 发送JSON字符串
            client_socket.sendall(message_str.encode('utf-8'))
        except socket.error as e:
            traceback.print_exc()
            # 处理发送数据时的错误
            # print(f"Socket error: {e}")


if __name__ == "__main__":
    server = BackendServer()
    sever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '127.0.0.1'
    port = 8886
    sever_socket.bind((host, port))
    sever_socket.listen()

    try:
        while True:
            conn, addr = sever_socket.accept()
            thread = threading.Thread(target=server.handle_connection, args=(conn, addr))
            thread.start()
    except Exception as e:
        # print("Server error:", e)
        traceback.print_exc()

    finally:
        sever_socket.close()
