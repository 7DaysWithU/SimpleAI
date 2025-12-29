"""
SimpleAI Framework
MIT License - Copyright (c) 2025 7DaysWithU@GitHub
"""

import os
import sys
import argparse
import warnings
import traceback
from datetime import datetime

from flask import Flask, request, jsonify
import eventlet
from eventlet import wsgi
from typing import Any

from control import GeneralDispatch
from utils.task import TaskManager

warnings.filterwarnings('ignore', category = FutureWarning)

app = Flask(__name__)
run = GeneralDispatch()

task_manager = TaskManager(max_workers = 2)


@app.route('/api/SimpleAI/config', methods = ['POST'])
def config() -> Any:
    """
    配置服务参数, 每次修改 settings.json文件就都需要重新调用 config()以加载新配置

    :return: 状态体
    :return: 状态码
    """

    try:
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, '../config/settings.json')

        # DEBUG:   '../config/settings.json'
        # PRODUCT: 'main/../config/settings.json'
        run.configure(config_path)

    except Exception:
        return jsonify({'error': 'Errors occurred during configuring', 'message': str(traceback.format_exc())}), 500

    else:
        return '', 204


@app.route('/api/SimpleAI/train', methods = ['POST'])
def train() -> Any:
    """
    训练模型

    :return: 状态体
    :return: 状态码
    """

    def str_to_bool(arg):
        """
        将 str转换为 bool

        :param arg: 待转换 str
        :return: bool
        """

        if arg == 'true':
            return True
        elif arg == 'false':
            return False
        else:
            return None

    try:
        incremental = request.args.get('incremental', type = str_to_bool)
        task_id = task_manager.submit(train_task, incremental)

    except Exception:
        return jsonify({'error': 'Errors occurred during training', 'message': str(traceback.format_exc())}), 500

    else:
        return jsonify({'task_id': task_id}), 202


@app.route('/api/SimpleAI/train/status/<task_id>', methods = ['GET'])
def train_status(task_id: str):
    """
    查询训练任务状态

    :param task_id: 任务 id
    :return: 返回任务的当前状态和结果
    """

    try:
        task = task_manager.result(task_id)

    except Exception:
        return jsonify({'error': f"Error occurred while querying task '{task_id}' status", 'message': str(traceback.format_exc())}), 500

    else:
        response = {
            'task_id': task.task_id,
            'state': task.state.name,
            'info': task.info
        }
        return jsonify(response), task.state.value[0]


@task_manager.long_task
def train_task(incremental: bool) -> Any:
    """
    训练模型

    :param incremental: 是否增量训练
    :return: 状态体
    :return: 状态码
    """

    with run:
        run.train(incremental)
        run.load_model()


@app.route('/api/SimpleAI/load', methods = ['POST'])
def load_model_into_memory() -> Any:
    """
    加载模型

    :return: 状态体
    :return: 状态码
    """

    try:
        run.load_model()

    except Exception:
        return jsonify({'error': 'Errors occurred during loading into memory', 'message': str(traceback.format_exc())}), 500

    else:
        return '', 204


@app.route('/api/SimpleAI/predict/<float:X>', methods = ['GET'])
def predict(X: int) -> Any:
    """
    获得預測值

    :return: 輸入值X
    :return: 輸出值y
    """

    try:
        with run:
            result = run.use([X]).tolist()

    except Exception:
        return jsonify({'error': 'Errors occurred during recommending to personal', 'message': str(traceback.format_exc())}), 500

    else:
        return jsonify(result), 200


def opening_show() -> None:
    """
    启动动画

    :return: 无
    """

    icon = r"""
       .-'''-.  .-./`)  ,---.    ,---. .-------.    .---.          .-''-.              ____     .-./`)  
      / _     \ \ .-.') |    \  /    | \  _(`)_ \   | ,_|        .'_ _   \           .'  __ `.  \ .-.') 
     (`' )/`--' / `-' \ |  ,  \/  ,  | | (_ o._)| ,-./  )       / ( ` )   '         /   '  \  \ / `-' \ 
    (_ o _).     `-'`"` |  |\_   /|  | |  (_,_) / \  '_ '`)    . (_ o _)  |         |___|  /  |  `-'`"` 
     (_,_). '.   .---.  |  _( )_/ |  | |   '-.-'   > (_)  )    |  (_,_)___|            _.-`   |  .---.  
    .---.  \  :  |   |  | (_ o _) |  | |   |      (  .  .-'    '  \   .---.   _ _   .'   _    |  |   |  
    \    `-'  |  |   |  |  (_,_)  |  | |   |       `-'`-'|___   \  `-'    /  ( ` )  |  _( )_  |  |   |  
     \       /   |   |  |  |      |  | /   )        |        \   \       /  (_{;}_) \ (_ o _) /  |   |  
      `-...-'    '---'  '--'      '--' `---'        `--------`    `'-..-'    (_,_)   '.(_,_).'   '---'  
    
    Welcome to Simple.AI
    Copyright (c) 2025 7DaysWithU@GitHub                                                                                         
    """

    print(f"[{datetime.now()}] * SimpleAI is starting")
    print(f"[{datetime.now()}] * SimpleAI version=1.0")
    print(icon)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Run Flask app with custom host and port")
    parser.add_argument('--host', type = str, default = '127.0.0.1', help = "Set the host (default: 127.0.0.1)")
    parser.add_argument('--port', type = int, default = 5000, help = "Set the port (default: 5000)")

    opening_show()

    args = parser.parse_args()
    eventlet.wsgi.server(eventlet.listen((args.host, args.port)), app)
