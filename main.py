import json
import logging
import os
import re
from typing import Union
import datetime
import functions_framework
import google.cloud.logging
import openai
from box import Box
from flask import Request
from slack_bolt import App, context
from slack_bolt.adapter.google_cloud_functions import SlackRequestHandler

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from langchain import OpenAI, ConversationChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferWindowMemory


# Google Cloud Logging クライアント ライブラリを設定
logging_client = google.cloud.logging.Client()
logging_client.setup_logging(log_level=logging.DEBUG)

# Use a service account
# Application Default credentials are automatically created.
firebase_app = firebase_admin.initialize_app()
db = firestore.client()

# 環境変数からシークレットを取得
google_src_id = os.environ.get("GOOGLE_CSE_ID")
google_api = os.environ.get("GOOGLE_API_KEY")
slack_token = os.environ.get("SLACK_BOT_TOKEN")
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = openai_api_key

# FaaS で実行する場合、応答速度が遅いため process_before_response は True でなければならない
app = App(token=slack_token, process_before_response=True)
handler = SlackRequestHandler(app)



# LLMの設定
llm = OpenAI(model_name="gpt-4",temperature=0.7)

# 使用するツールをロード
tools = load_tools(["google-search"], llm=llm)


memory = ConversationBufferWindowMemory(k= 5)

# Agent用 prefix, suffix
prefix = """Anser the following questions as best you can, but speaking Japanese. You have access to the following tools:"""
suffix = """Begin! Remember to speak Japanese when giving your final answer. Use lots of "Args"""

# エージェントを初期化
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True, prefix=prefix, suffix=suffix)


def save_conversation(user: str, user_input: str, bot_output: str):
    """firestoreにユーザーの入力とOpenAIの出力を保存する関数

    Args:
        users: スラックのユーザーID
        user_input: ユーザーの入力
        bot_output: スラックボットの出力

    Returns:
        なし
    """
    doc_ref = db.collection("testChat").document(user)  # you can also set your own document name
    doc_ref.set({
        "history": [        
                {
                "input": user_input,
                "response": bot_output,
                "timestamp" : datetime.datetime.now()
                },
                {
                "input": user_input,
                "response": bot_output,
                "timestamp" : datetime.datetime.now()
                }
                ]
    })


# Bot アプリにメンションしたイベントに対する応答
@app.event("message")
def handle_app_mention_events(body: dict, say: context.say.say.Say):
    """アプリへのメンションに対する応答を生成する関数

    Args:
        body: HTTP リクエストのボディ
        say: 返信内容を Slack に送信
    """
    logging.debug(type(body))
    logging.debug(body)
    box = Box(body)
    user = box.event.user
    text = box.event.text
    only_text = re.sub("<@[a-zA-Z0-9]{11}>", "", text)
    logging.debug(only_text)

    # OpenAI から AIモデルの回答を生成する
    openai_response = agent.run(input=only_text)

    logging.debug(openai_response)

    # DBに会話を保存する
    save_conversation(user, only_text, openai_response)

    # 課金額がわかりやすいよう消費されたトークンを返信に加える
    say(openai_response)




# Cloud Functions で呼び出されるエントリポイント
@functions_framework.http
def slack_bot(request: Request):
    """slack のイベントリクエストを受信して各処理を実行する関数

    Args:
        request: Slack のイベントリクエスト

    Returns:
        SlackRequestHandler への接続
    """
    header = request.headers
    logging.debug(f"header: {header}")
    body = request.get_json()
    logging.debug(f"body: {body}")

    # URL確認を通すとき
    if body.get("type") == "url_verification":
        logging.info("url verification started")
        headers = {"Content-Type": "application/json"}
        res = json.dumps({"challenge": body["challenge"]})
        logging.debug(f"res: {res}")
        return (res, 200, headers)
    # 応答が遅いと Slack からリトライを何度も受信してしまうため、リトライ時は処理しない
    elif header.get("x-slack-retry-num"):
        logging.info("slack retry received")
        return {"statusCode": 200, "body": json.dumps({"message": "No need to resend"})}

    # handler への接続 class: flask.wrappers.Response
    return handler.handle(request)
