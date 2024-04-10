import base64
import traceback
from concurrent.futures import ThreadPoolExecutor
import socket
import json
import numpy as np
import cv2
IP = '127.0.0.1'
port = 8886
def send_msg(dict1):
    print(1)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP, port))
    s.send(f"msg".encode())
    print(f"{dict1}")
    data = json.dumps(dict1)
    # data = dict1.encode()
    s.sendall(data.encode())
    print("msg sent successfully")
    s.close()
    # json_data = json.dumps(dict_data)
    # c_socket.sendall(f"msg".encode())
    # c_socket.sendall(json_data.encode())

def send_msg1(c_client,test_data):
    print(4)
    c_client.send(json.dumps(test_data).encode('utf-8'))
    print("send successfully!")
    # c_client.close()

def send_img(img):
    print(2)
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((IP, port))
        s.send(f"img".encode())
        s.sendall(img_base64)
        print("Image sent successfully.")
    except Exception as e:
        traceback.print_exc()

    finally:
        # 关闭socket连接
        s.close()
    
    

    # c_socket.sendall(f"img".encode())
    # c_socket.sendall(img)

def recv_msg(c_socket):
    print(3)
    recv_header = c_socket.recv(3).decode('utf-8')

    if recv_header == "msg":
        buffer_size = 1024
        data_str = ""
        while True:
            part = c_socket.recv(buffer_size).decode('utf-8')
            data_str += part
            if len(part) < buffer_size:
                # 假设数据已全部接收
                break
            # 尝试解析接收到的数据为字典
        try:
            data = json.loads(data_str)
            return data  # 返回解析后的字典
        except json.JSONDecodeError:
            traceback.print_exc()

            return None

def recv_img(c_socket):
    print(4)
    recv_header = c_socket.recv(3).decode('utf-8')
    if recv_header == "img":
        buffer_size = 1024
        img_data = b''
        while True:
            packet = c_socket.recv(buffer_size)
            if len(packet) < buffer_size:
                break
            img_data += packet
    
    return img_data

def send_json_msg(message):
        """
        向客户端发送JSON格式的消息。
        :param client_socket: 客户端socket对象
        :param message: 要发送的消息（字典）
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((IP, port))
        try:
            # 将消息字典转换为JSON字符串
            message_str = json.dumps(message)
            # 发送JSON字符串
            s.sendall(message_str.encode('utf-8'))
        except socket.error as e:
            traceback.print_exc()
            # 处理发送数据时的错误
            # print(f"Socket error: {e}")
        finally:
            s.close()

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
        traceback.print_exc()
        # 如果解析JSON失败，抛出异常
        # print(f"JSON decode error: {e}")
        return None

if __name__ == "__main__":
    # test_data = {"type": "test", "content": "This is a test message."}
    # send_msg(test_data)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0]
    send_img(img)

    

