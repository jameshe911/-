# -*- coding: utf-8 -*-
from openai import OpenAI

client = OpenAI(
    api_key="sk-c6572a0228a2402fb29907ca425a95d0",
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """你是一个刚收到货的普通买家。用一两句话说出最真实的感受，像发朋友圈吐槽一样自然。最后附一行JSON：{"sentiment": "正面/负面/中性", "confidence": 0.0~1.0}"""

print("=== 情感分析 ===\n")

while True:
    text = input("商品: ")
    if text == 'q':
        break

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )

    print(response.choices[0].message.content)
    print()