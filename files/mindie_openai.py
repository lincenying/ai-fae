import time
import requests

# 初始化对话历史
dialog_history = []


def send_request_with_token_management(user_question):
    global dialog_history

    def calculate_tokens(history):
        """计算历史记录中的总tokens数量"""
        total_tokens = sum(len(message["content"]) for message in history)
        return total_tokens

    # 将用户问题添加到对话历史中
    dialog_history.append({"role": "user", "content": user_question})

    # 计算历史记录总tokens
    total_tokens = calculate_tokens(dialog_history)

    # 如果总tokens接近或超过max_tokens，则清空历史记录
    if total_tokens >= 8192:
        dialog_history.clear()

    url = "http://39.171.244.84:40002/v1/chat/completions"
    api_key = "1"
    headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    data = {
        "model": "qwen1.5-72b",
        "max_tokens": 1024,
        "presence_penalty": 1.03,
        "frequency_penalty": 1.0,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.95,
        "stream": False,
        "messages": [{"role": "system", "content": "You are a helpful assistant."}] + dialog_history,  # 添加所有历史对话
    }

    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result_json = response.json()
            content = result_json["choices"][0]["message"]["content"]
            print("返回内容:", content)
            completion_tokens = result_json["usage"]["completion_tokens"]
            end_time = time.time()
            total_time = end_time - start_time
            inference_speed = completion_tokens / total_time

            # 将AI的回答添加到对话历史中
            dialog_history.append({"role": "assistant", "content": content})

            return content, total_time, inference_speed
        else:
            print(f"请求失败，状态码：{response.status_code}")
            print("响应内容：", response.text)
    except Exception as e:
        print(f"请求过程中发生错误：{e}")


if __name__ == "__main__":
    while True:
        user_question = input("请输入您的问题(输入'exit'退出程序): ")
        if user_question.lower() == "exit":
            break
        send_request_with_token_management(user_question)
