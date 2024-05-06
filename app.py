import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from main import run, add_to_queue, gradio_interface
import io
import sys
import time
import os
import pandas as pd
OPENAI_KEY = "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"
# OPENAI_KEY = None

# 定义CSS样式多行字符串，用于自定义Gradio界面的外观
css = """#col-container {max-width: 90%; margin-left: auto; margin-right: auto; display: flex; flex-direction: column;}
#header {text-align: center;}
#col-chatbox {flex: 1; max-height: min(750px, 100%);}
#label {font-size: 4em; padding: 0.5em; margin: 0;}
.scroll-hide {overflow-y: scroll; max-height: 100px;}
.wrap {max-height: 680px;}
.message {font-size: 3em;}
.message-wrap {max-height: min(700px, 100vh);}
body {
        background-color: #ADD8E6;
    }
"""

# 配置Matplotlib以支持特定的字体和设置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK']
plt.rcParams['axes.unicode_minus'] = False

# 示例查询
example_queries = [
    '我想了解人脑神经元细胞的生产情况',
    '我要查看手动重建数据的特征分布',
    '手动重建和自动重建的数据版本有什么区别',
    '人工标注与自动重建的神经元特征分布有什么不同',
    '看一下三类图像数据的质量对比',
    '给我画出人脑神经元的生产趋势图',
    '图像数据有几种格式?分别是什么?'
]

# 定义'Clinet'类，处理数据请求和与OpenAI API交互的
class Client:
    def __init__(self) -> None:
        # API相关变量初始化
        self.OPENAI_KEY = OPENAI_KEY
        self.OPENAI_API_BASED_AZURE = None
        self.OPENAI_ENGINE_AZURE = None
        self.OPENAI_API_KEY_AZURE = None
        self.stop = False  # 添加停止标志，用于控制查询处理循环
    
    def set_key(self, openai_key, openai_key_azure, api_base_azure, engine_azure):
        # 设置和获取API密钥和相关配置
        self.OPENAI_KEY = openai_key
        self.OPENAI_API_BASED_AZURE = api_base_azure
        self.OPENAI_API_KEY_AZURE = openai_key_azure
        self.OPENAI_ENGINE_AZURE = engine_azure
        return self.OPENAI_KEY, self.OPENAI_API_KEY_AZURE, self.OPENAI_API_BASED_AZURE, self.OPENAI_ENGINE_AZURE

    def run(self, messages):
        # 根据设置的API密钥执行查询，并处理生成的数据
        if self.OPENAI_KEY == '' and self.OPENAI_API_KEY_AZURE == '':
            yield '', np.zeros((100, 100, 3), dtype=np.uint8), "Please set your OpenAI API key first!!!", pd.DataFrame()
        elif len(self.OPENAI_KEY) >= 0 and not self.OPENAI_KEY.startswith('sk') and self.OPENAI_API_KEY_AZURE == '':
            yield '', np.zeros((100, 100, 3), dtype=np.uint8), "Your openai key is incorrect!!!", pd.DataFrame()
        else:
            # self.stop = False
            gen = gradio_interface(messages, self.OPENAI_KEY, self.OPENAI_API_KEY_AZURE, self.OPENAI_API_BASED_AZURE, self.OPENAI_ENGINE_AZURE)
            while not self.stop:  #
                try:
                    yield next(gen)
                except StopIteration:
                    print("StopIteration")
                    break

# 使用'gr.Blocks'来构建Gradio的界面，包括文本输入框、按钮、下拉菜单和输出显示区域等组件，以及这些组件的布局和样式设置
with gr.Blocks() as demo:
    state = gr.State(value={"client": Client()})    # 使用'gr.State'来保存整个应用的状态，比如用户的API密钥设置

    # 根据用户的选择更新查询输入框
    def change_textbox(query):
        # 根据不同输入对输出控件进行更新
        return gr.update(lines=2, visible=True, value=query)
    # 图片框显示
    
    with gr.Row():
        gr.Markdown(
        """
        # Hello Human Neuron Data Explorer! 🧠
        A powerful AI system connects humans and data.
        Explore the mysteries of human brain neurons.

        
        Please enter your question and click 'start'! 
        """)
    
    # 如果OpenAI的API密钥未设置，则显示输入框供用户输入
    if not OPENAI_KEY:
        with gr.Row().style():
            with gr.Column(scale=0.9):
                gr.Markdown(
                    """
                    You can use gpt35 from openai or from openai-azure.
                    """)
                openai_api_key = gr.Textbox(
                    show_label=False,
                    placeholder="Set your OpenAI API key here and press Submit  (e.g. sk-xxx)",
                    lines=1,
                    type="password"
                ).style(container=False)

                with gr.Row():
                    openai_api_key_azure = gr.Textbox(
                        show_label=False,
                        placeholder="Set your Azure-OpenAI key",
                        lines=1,
                        type="password"
                    ).style(container=False)
                    openai_api_base_azure = gr.Textbox(
                        show_label=False,
                        placeholder="Azure-OpenAI api_base (e.g. https://zwq0525.openai.azure.com)",
                        lines=1,
                        type="password"
                    ).style(container=False)
                    openai_api_engine_azure = gr.Textbox(
                        show_label=False,
                        placeholder="Azure-OpenAI engine here (e.g. gpt35)",
                        lines=1,
                        type="password"
                    ).style(container=False)


                gr.Markdown(
                    """
                    It is recommended to use the Openai paid API or Azure-OpenAI service, because the free Openai API will be limited by the access speed and 3 Requests per minute (very slow).
                    """)

            with gr.Column(scale=0.1, min_width=0):
                btn1 = gr.Button("OK").style(height= '100px')
    
    # with gr.Row():
    #     input_text = gr.inputs.Textbox(lines=1, placeholder='Enter your neuron data query...', label='What neuron data are you interested in?')
    #     start_btn = gr.Button("Start").style(full_height=True)
    
    with gr.Row():
        with gr.Column(scale=0.9):
            input_text = gr.inputs.Textbox(lines=1, placeholder='Please input your problem...', label='what do you want to find?')

        with gr.Column(scale=0.1, min_width=0):
            start_btn = gr.Button("Start").style(full_height=True)
            # end_btn = gr.Button("Stop").style(full_height=True)
            
    gr.Markdown(
        """
        # Try these neuron data queries ➡️➡️
        """)
    with gr.Row():
        example_selector = gr.Dropdown(choices=example_queries, interactive=True,
                                        label="Neuron Data Queries:", show_label=True)
    
    def run(state, query):
        # 根据用户查询执行神经元数据分析
        generator = state["client"].run(query)
        for solving_step, img, res, df in generator:
            yield solving_step, img, res, df

    # 设置输出部分，展示分析结果和相关数据
    # with gr.Row():
    #     Res = gr.Textbox(label="Summary and Result: ")
    #     solving_step = gr.Textbox(label="Analysis Step: ", lines=5)
            
    with gr.Row():
        with gr.Column(scale=0.3, min_width="500px", max_width="500px", min_height="500px", max_height="500px"):
                Res = gr.Textbox(label="Summary and Result: ")
        with gr.Column(scale=0.7, min_width="500px", max_width="500px", min_height="500px", max_height="500px"):
            solving_step = gr.Textbox(label="Solving Step: ", lines=5)
    
    img = gr.outputs.Image(type='numpy')
    df = gr.outputs.Dataframe(type='pandas')
    
    with gr.Row():
        gr.Markdown(
            """
            [OpenAI](https://openai.com/) provides the powerful Chatgpt model for our Data Explorer.

            """)
    
    outputs = [solving_step, img, Res, df]

    #设置change事件
    example_selector.change(fn = change_textbox, inputs=example_selector, outputs=input_text)
    
    start_btn.click(fn=run, inputs=[state, input_text], outputs=outputs)
    
    # 启动队列和应用
    demo.queue()
    demo.launch()
    
