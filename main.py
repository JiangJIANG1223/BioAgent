import matplotlib.pyplot as plt
# import matplotlib.figure as fgr
import pandas as pd
import os
import json
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from lab_gpt4_call import send_chat_request,send_chat_request_Azure,send_official_call
from lab_llms_call import send_chat_request_qwen,send_chat_request_glm,send_chat_request_chatglm3_6b,send_chat_request_chatglm_6b
# from lab_llm_local_call import send_chat_request_internlm_chat
#import ast
import re
from tool import *
import tiktoken
import concurrent.futures
from PIL import Image
from io import BytesIO
import  queue
# import datetime
from threading import Thread
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False
import openai
import time

# To override the Thread method
# 用于并发执行任务，并获取任务结果
class MyThread(Thread):

    def __init__(self, target, args):
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result

# 解析给定的任务字典，执行对应的函数，并将结果存储在结果缓存区中（不接受参数版）
def parse_and_exe1(call_dict, result_buffer, parallel_step: str='1'):
    """
    Parse the input and call the corresponding function to obtain the result.
    :param call_dict: dict, now only including func, output, and desc
    :param result_buffer: dict, storing the corresponding intermediate results
    :param parallel_step: str, parallel step
    :return: Returns func() and stores the corresponding result in result_buffer.
    """
    func_name = call_dict['function' + parallel_step]
    output = call_dict['output' + parallel_step]
    desc = call_dict['description' + parallel_step]
    
    # 假设所有函数都不接受参数
    result = eval(func_name)()
    
    # 存储结果和描述
    result_buffer[output] = (result, desc)
    
    return result_buffer

# 解析给定的任务字典，执行对应的函数，并将结果存储在结果缓存区中（接受参数版）
def parse_and_exe2(call_dict, result_buffer, parallel_step: str='1'):
    """
    Parse the input and call the corresponding function to obtain the result.
    :param call_dict: dict, including arg, func, and output
    :param result_buffer: dict, storing the corresponding intermediate results
    :param parallel_step: int, parallel step
    :return: Returns func(arg) and stores the corresponding result in result_buffer.
    """
    arg_list = call_dict['arg' + parallel_step]
    replace_arg_list = [result_buffer[item][0] if isinstance(item, str) and ('result' in item or 'input' in item) else item for item in arg_list]  # 参数
    func_name = call_dict['function' + parallel_step]             
    output = call_dict['output' + parallel_step]                  
    desc = call_dict['description' + parallel_step]               
    if func_name == 'loop_rank':
        replace_arg_list[1] = eval(replace_arg_list[1])
    result = eval(func_name)(*replace_arg_list)
    result_buffer[output] = (result, desc)    #    'result1': (df1, desc)
    return result_buffer

# 读取tool lib 和 tool prompt，将它们整合为一个扁平化的提示字符串
# 将工具库的描述和提示信息整合在一起，形成一个统一的、易于阅读和使用的提示字符串
def load_tool_and_prompt(tool_lib, tool_prompt ):
    '''
    Read two JSON files.
    :param tool_lib: Tool description
    :param tool_prompt: Tool prompt
    :return: Flattened prompt
    '''

    with open(tool_lib, 'r', encoding='utf-8') as f:
        tool_lib = json.load(f)

    with open(tool_prompt, 'r', encoding='utf-8') as f:
        tool_prompt = json.load(f)

    # 遍历'tool_lib'字典中的所有项，将每个工具函数的描述添加到'tool_prompt'字典中的'Function:'键下
    # 这个过程将工具库描述整合到提示信息中，每个工具及其描述之间用空行隔开('\n\n')
    for key, value in tool_lib.items():
        tool_prompt["Function Library:"] = tool_prompt["Function Library:"] + key + " " + value+ '\n\n'

    # 遍历'tool_prompt'字典中的所有项，将每个键值对追加到'prompt_flat'字符串中，每对之间也用空行隔开（'\n\n'）
    # 这一步将所有提示信息整合成一个扁平化的字符串，方便后续使用。
    prompt_flat = ''
    for key, value in tool_prompt.items():
        prompt_flat = prompt_flat + key +'  '+ value + '\n\n'

    return prompt_flat

# callback function
intermediate_results = queue.Queue()  # Create a queue to store intermediate results.

# 将中间结果添加到队列中，以便近一步处理
def add_to_queue(intermediate_result):
    intermediate_results.put(f"After planing, the intermediate result is {intermediate_result}")

# 限制请求的频率，以符合某些API的速率限制要求
def check_RPM(run_time_list, new_time, max_RPM=1):
    # Check if there are already 3 timestamps in the run_time_list, with a maximum of 3 accesses per minute.
    # False means no rest is needed, True means rest is needed.
    if len(run_time_list) < 3:
        run_time_list.append(new_time)
        return 0
    else:
        if (new_time - run_time_list[0]).seconds < max_RPM:
            # Calculate the required rest time.
            sleep_time = 60 - (new_time - run_time_list[0]).seconds
            print('sleep_time:', sleep_time)
            run_time_list.pop(0)
            run_time_list.append(new_time)
            return sleep_time
        else:
            run_time_list.pop(0)
            run_time_list.append(new_time)
            return 0

### 主要的业务逻辑函数，解析指令、规划任务、选择工具、并执行生成最终结果的步骤
def run(instruction, add_to_queue=None, send_chat_request_Azure = send_official_call, openai_key = '', api_base='', engine=''):
    '''
    :param instruction: 用户提供的指令/查询
    :param add_to_queue: 一个回调函数，用于将中间结果添加到队列中，实时反馈给用户
    :param send_chat_request_Azure: 用于发送请求到Azure版本的OpenAI API的函数
    :openai_key, api_base, engine 与OpenAI API相关的配置参数
    :return: output_text, image, output_result, df
    '''
    
    output_text = ''    # 初始化输出文本为空字符串

    ## 意图检测，通过读取意图检测的提示库('prompt_intent_detection.json')，结合用户提供的指令，生成一个新的指令('prompt_intent_detection')
    ################################# Step-1:Task select ###########################################
    print('===============================Intent Detecting===========================================')
    with open('./prompt_lib/prompt_intent_detection.json', 'r', encoding='utf-8') as f:
        prompt_task_dict = json.load(f)   # 加载意图检测的提示库
    prompt_intent_detection = ''
    for key, value in prompt_task_dict.items():  # 遍历提示库，构建意图检测指令
        prompt_intent_detection = prompt_intent_detection + key + ": " + value+ '\n\n'

    prompt_intent_detection = prompt_intent_detection + '\n\n' + 'Instruction:' + instruction + ' ###New Instruction: '
    
    ## 将指令('prompt_intent_detection')发送给语言模型，转化为一个更加明确的任务描述('new_instruction')
    response = send_chat_request("gpt", prompt_intent_detection, openai_key=openai_key, api_base=api_base, engine=engine)

    new_instruction = response  # 将响应内容作为新的指令
    print('new_instruction:', new_instruction)
    output_text = output_text + '\n======Intent Detecting Stage=====\n\n'
    output_text = output_text + new_instruction +'\n\n'

    if add_to_queue is not None:
        add_to_queue(output_text)  # 队列回调函数，将输出文本添加到队列中

    event_happen = True  # 时间发生标志位


    ## 将用户的意图转化为一系列具体的任务步骤，每个步骤都有相应的工具和方法来执行
    ## 读取任务规划的提示库('prompt_task.json'), 并将意图检测的结果('new_instruction')作为输入，生成任务规划指令('prompt_task')
    print('===============================Task Planing===========================================')
    output_text= output_text + '=====Task Planing Stage=====\n\n'

    with open('./prompt_lib/prompt_task.json', 'r', encoding='utf-8') as f:
        prompt_task_dict = json.load(f)  # 加载任务规划的提示库
    prompt_task = ''
    for key, value in prompt_task_dict.items():  # 遍历提示库，构建任务规划指令
        prompt_task = prompt_task + key + ": " + value+ '\n\n'  

    prompt_task = prompt_task + '\n\n' + 'Instruction:' + new_instruction + ' ###Plan:'
    
    # 发送任务规划指令('prompt_task')到语言模型，并获取响应
    response = send_chat_request("gpt", prompt_task, openai_key=openai_key, api_base=api_base, engine=engine)

    task_select = response  # 任务选择结果
    
    pattern = r"(task\d+=)(\{[^}]*\})"  # 使用正则表达式匹配任务
    matches = re.findall(pattern, task_select)  # 查找所有匹配的任务
    task_plan = {}  # 初始化任务计划字典
    for task in matches:
        task_step, task_select = task
        task_select = task_select.replace("'", "\"")  # Replace single quotes with double quotes.
        task_select = json.loads(task_select)  # 解析JSON格式的任务描述
        task_name = list(task_select.keys())[0]  # 获取任务名称
        task_instruction = list(task_select.values())[0]  # 获取任务指令

        task_plan[task_name] = task_instruction  # 将任务添加到任务计划字典中

    # 遍历任务计划，打印每个任务的名称和指令
    # task_plan
    for key, value in task_plan.items():
        print(key, ':', value)
        output_text = output_text + key + ': ' + str(value) + '\n'

    output_text = output_text +'\n'
    if add_to_queue is not None:
        add_to_queue(output_text)  # 将输出文本添加到队列中


    ## 根据任务规划的结果，为每个任务选择合适的工具
    ################################# Step-2:Tool select and use ###########################################
    print('===============================Tool select and using Stage===========================================')
    output_text = output_text + '======Tool select and using Stage======\n\n'
    
    # Read the task_select JSON file name.
    # task_name = list(task_plan.keys())[0].split('_task')[0]
    task_instruction = list(task_plan.values())[0]

    tool_lib = './tool_lib/tool_loading.json'    # 工具库文件
    tool_prompt = './prompt_lib/prompt_loading.json'  # 工具提示文件路径
    prompt_flat = load_tool_and_prompt(tool_lib, tool_prompt)  # 加载工具和提示
    prompt_flat = prompt_flat + '\n\n' +'Instruction :'+ task_instruction+ ' ###Function Call'
    
    response = send_chat_request("gpt", prompt_flat, openai_key=openai_key,api_base=api_base, engine=engine)

    call_steps, _ = response.split('###')  # 分割响应内容，获取调用步骤
    pattern = r"(step\d+=)(\{[^}]*\})"  # 使用正则表达式匹配步骤
    matches = re.findall(pattern, call_steps)  # 查找所有匹配的步骤 每一个匹配项都是一个元组('step1=' 和 步骤的JSON描述)
    
    # 初始化结果缓冲区和输出缓冲区
    # result_buffer = {} 用于存储每个步骤的执行结果; output_buffer = [] 用于记录最终输出的变量名
    result_buffer = {}    # The stored format is as follows: {'result1': (000001.SH, 'Stock code of China Ping An'), 'result2': (df2, 'Stock data of China Ping An from January to June 2021')}.
    output_buffer = []    # Store the variable names [result5, result6] that will be passed as the final output to the next task.

    for match in matches:  # 遍历所有匹配的步骤
        step, content = match
        content = content.replace("'", "\"")  # Replace single quotes with double quotes.
        
        # 输出当前步骤的信息，包括步骤标识和内容
        print('==================')
        print("\n\nstep:", step)
        print('content:',content)  

        # 计算并打印出每个步骤中包含的并行子步骤数量（每个子步骤占用3个字段：函数名、输出变量、描述）
        call_dict = json.loads(content)  # 解析JSON格式的步骤描述
        print('It has parallel steps:', len(call_dict) / 3)  
        output_text = output_text + step + ': ' + str(call_dict) + '\n\n'

        # 使用多线程并行执行步骤
        # Execute the following code in parallel using multiple processes.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to thread pool 提交任务到线程池
            futures = {executor.submit(parse_and_exe1, call_dict, result_buffer, str(parallel_step))
                       for parallel_step in range(1, int(len(call_dict) / 3) + 1)}

            # Collect results as they become available 收集结果
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                # Handle possible exceptions 处理可能的异常
                try:
                    result = future.result()
                    # Print the current parallel step number. 打印当前并行步骤编号
                    print('parallel step:', idx+1)
                    # print(list(result[1].keys())[0])
                    # print(list(result[1].values())[0])
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        if step == matches[-1][0]:  # 如果是当前任务的最后一个步骤，保存最终步骤的输出
            # Current task's final step. Save the output of the final step.
            for parallel_step in range(1, int(len(call_dict) / 3) + 1):
                output_buffer.append(call_dict['output' + str(parallel_step)])
    
    output_text = output_text + '\n'
    
    print(f"result_buffer: {result_buffer}")
    print(len(result_buffer))

    print(f"output_buffer: {output_buffer}")

    if add_to_queue is not None:
        add_to_queue(output_text)


    ## 根据任务规划阶段确定的可视化任务，选择和执行相应的可视化工具
    ## 读取工具库和提示信息，生成一个函数调用的指令
    # 解析和执行可视化指令，生成最终的图表或数据表
    ################################# Step-3:visualization ###########################################
    print('===============================Visualization Stage===========================================')
    output_text = output_text + '======Visualization Stage====\n\n'
    
    task_instruction = list(task_plan.values())[1]  # 可视化任务指令

    tool_lib = './tool_lib/tool_visualization.json'    # 可视化工具库文件
    tool_prompt = './prompt_lib/prompt_visualization.json'  # 可视化工具提示文件路径

    # 初始化可视化结果缓冲区和前置结果字典
    result_buffer_viz = {}
    Previous_result = {}

    # 遍历输出缓冲区
    for output_name in output_buffer:
        rename = 'input'+ str(output_buffer.index(output_name)+1)
        Previous_result[rename] = result_buffer[output_name][1]
        result_buffer_viz[rename] = result_buffer[output_name]

    prompt_flat = load_tool_and_prompt(tool_lib, tool_prompt)  # 加载可视化工具和提示
    prompt_flat = prompt_flat + '\n\n' +'Instruction: '+ task_instruction + ', Previous_result: '+ str(Previous_result) + ' ###Function Call'

    # 发送可视化指令到语言模型，并获取响应
    response = send_chat_request("gpt", prompt_flat, openai_key=openai_key, api_base=api_base, engine=engine)

    call_steps, _ = response.split('###')  # 分割相应内容，获取调用步骤
    pattern = r"(step\d+=)(\{[^}]*\})"  # 使用正则表达式匹配步骤
    matches = re.findall(pattern, call_steps)  # 查找所有匹配的步骤
    
    for match in matches:  # 遍历所有匹配的步骤
        step, content = match
        content = content.replace("'", "\"")  # Replace single quotes with double quotes.
        print('==================')
        print("\n\nstep:", step)
        print('content:', content) 
        call_dict = json.loads(content)  # 解析JSON格式的步骤描述
        print('It has parallel steps:', len(call_dict) / 4)
        result_buffer_viz = parse_and_exe2(call_dict, result_buffer_viz, parallel_step = '' )
        output_text = output_text + step + ': ' + str(call_dict) + '\n\n'

    if add_to_queue is not None:
        add_to_queue(output_text)

    # 最终的可视化结果
    finally_output = list(result_buffer_viz.values())  # plt.Axes

    for index, obj in enumerate(finally_output):
        print(f"对象索引：{index}, 类型：{type(obj[0])}")

    df = pd.DataFrame()  # 初始化一个空的DataFrame
    str_out = output_text + 'Finally result: '

    for ax in finally_output:  # 遍历可视化结果
        # 如果结果是plt.Axes, 显示它
        # if isinstance(ax[0], plt.Axes) or isinstance(ax[0], fgr.Figure):         # If the output is plt.Axes, display it.
        if isinstance(ax[0], plt.Axes):
            print("Here is a plt.Axes or Figure to display.")

            buf = BytesIO()
            # plt.savefig('VC_main.png')
            plt.savefig(buf, format='png')
            buf.seek(0)    # 重置指针位置
            image = Image.open(buf)
            
            image.show()

            # plt.grid()
            # plt.show()
            str_out = str_out + ax[1]+ ':' + 'plt.Axes' + '\n\n'

        # 如果结果是DataFrame, 记录它
        elif isinstance(ax[0], pd.DataFrame):
            print("Here is a pd.DataFrame to record.")

            df = ax[0]
            str_out = str_out + ax[1]+ ':' + 'pd.DataFrame' + '\n\n'

        else:
            str_out = str_out + str(ax[1])+ ':' + str(ax[0]) + '\n\n'


    ## 调用另一个语言模型来总结整个任务的执行过程和结果
    ## 将最终的文本结果、图表、数据表作为函数的输出
    print('===============================Summary Stage===========================================')
    output_prompt = "请用第一人称总结一下整个任务规划和解决过程,并且输出结果,用[Task]表示每个规划任务,用\{function\}表示每个任务里调用的函数." + \
                    "示例1:###我将您的问题拆分成两个任务,首先第一个任务[loading_task],我获取人脑神经元细胞的生产信息数据. \n然后第二个任务[visualization_task],我用折线图绘制人脑神经元细胞生产趋势. \n\n在第一个任务中我使用了1个工具函数\{get_production_data\}获取到人脑神经元的生产数据,在第二个任务里我们使用\{plot_production_trend\}工具函数来绘制趋势图. \n\n最后我们给您提供了人脑神经元细胞的生产趋势图和数据表格."+ \
                    "示例2:###我将您的问题拆分成两个任务,首先第一个任务[loading_task],我获取手动重建神经元的特征数据. \n然后第二个任务[visualization_task],我绘制手动重建数据的特征分布直方图和KDE. \n\n在第一个任务中我使用了1个工具函数\{get_manual_data\} 获取到手动重建神经元特征数据,在第二个任务里我们使用\{plot_feature_distribution\}工具函数来绘制特征分布直方图和KDE. \n\n最后我们给您提供了手动重建神经元数据的特征分布图和数据表格."+ \
                    "示例3:###我将您的问题拆分成两个任务,首先第一个任务[loading_task],我依次获取手动重建和自动重建的特征数据. \n第二个任务是[visualization_task],我在一张图里绘制手动重建数据和自动重建数据特征分布的小提琴图. \n\n为了完成第一个任务我分别使用了2个工具函数\{get_manual_data\},\{get_auto_data\}分别获取到手动重建神经元特征数据和自动重建神经元特征数据,第二个任务里我们使用绘图工具函数\{plot_version_comparison\}绘制两种数据的小提琴图.\n\n最后我们给您提供了包含两种数据的特征分布图和数据表格."
    # output_result = send_chat_request("qwen-chat-72b", output_prompt + str_out + '###')
    output_result = send_chat_request("gpt", output_prompt + str_out + '###', openai_key=openai_key, api_base=api_base,engine=engine)
    print(output_result)

    # buf = BytesIO()
    # plt.savefig(buf, format='png')
    # plt.savefig('StateofHumanBrainCells_main_test.png')

    # buf.seek(0)  # 重置指针位置

    # image = Image.open(buf)

    # try:
    #     image = Image.open(buf)
    #     # image.show()
    #     print("Image data is valid.")
    # except IOError:
    #     print("Image data is not valid.")

    return output_text, image, output_result, df


# 封装'run'函数的调用，使其可以通过Gradio界面异步运行，并返回实时的中间结果和最终结果
def gradio_interface(query, openai_key, openai_key_azure, api_base,engine):
    # Create a new thread to run the function.
    # if openai_key.startswith('sk') and openai_key_azure == '':
    #     print('send_official_call')
    #     thread = MyThread(target=run, args=(query, add_to_queue, send_official_call, openai_key))
    # elif openai_key =='' and len(openai_key_azure)>0:
    #     print('send_chat_request_Azure')
    #     thread = MyThread(target=run, args=(query, add_to_queue, send_chat_request_Azure, openai_key_azure, api_base, engine))

    thread = MyThread(target=run, args=(query, add_to_queue, send_official_call, openai_key))
    thread.start()
    placeholder_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Create a placeholder image.
    placeholder_dataframe =  pd.DataFrame()                      

    # Wait for the result of the calculate function and display the intermediate results simultaneously.
    while thread.is_alive():
        while not intermediate_results.empty():
            yield intermediate_results.get(), placeholder_image,  'Running' , placeholder_dataframe         # Use the yield keyword to return intermediate results in real-time
        time.sleep(0.1)                                          # Avoid excessive resource consumption.

    finally_text, img, output, df = thread.get_result()
    yield  finally_text, img, output, df
    # Return the final result.

# 发送请求到不同的语言模型并获取响应
def send_chat_request(model, prompt, send_chat_request_Azure = send_official_call, openai_key = '', api_base='', engine=''):
    '''
    Send request to LLMs(gpt, qwen-chat-72b, glm-3-turbo...)
    :param model: the name of llm
    :param prompt: prompt
    :param send_chat_request_Azure(for gpt call)
    :param openai_key(for gpt call)
    :param api_base(for gpt call)
    :param engine(for gpt call)
    :return response: the response of llm
    '''
    if model=="gpt":
        response = send_chat_request_Azure(prompt, openai_key=openai_key, api_base=api_base, engine=engine)
    elif model=="qwen-chat-72b":
        response = send_chat_request_qwen(prompt)# please set your api_key in lab_llms_call.py 
    elif model=="glm-3-turbo":
        response = send_chat_request_glm(prompt)# please set your api_key in lab_llms_call.py 
    elif model =="chatglm3-6b":
        response = send_chat_request_chatglm3_6b(prompt)# please set your api_key in lab_llms_call.py 
    elif model=="chatglm2-6b":
        response =  send_chat_request_chatglm_6b(prompt)# please set your api_key in lab_llms_call.py 
    # If you want to call the llm from local, you can try the following: internlm-chat-7b
    # elif model=="internlm-chat-7b":
    #     response = send_chat_request_internlm_chat(prompt)  
    return response


instruction = ''

if __name__ == '__main__':
    # 初始化pro接口
    # openai_call = send_chat_request_Azure #
    
    # if using gpt, please set the following parameters
    openai_call = send_official_call #
    openai_key = os.getenv("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")
    
    output, image, df , output_result = run(instruction, send_chat_request_Azure = openai_call, openai_key=openai_key, api_base='', engine='')
    print(output_result)
    plt.show()







