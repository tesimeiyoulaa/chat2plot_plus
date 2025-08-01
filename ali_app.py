import json
import logging
import os
import subprocess
import sys
import time
import traceback

import pandas as pd
import streamlit as st
from plotly.graph_objs import Figure
from pydantic import BaseModel
from streamlit_chat import message

# --- 修改 1: 导入正确的 ChatOpenAI ---
# 注意：根据警告信息，应从 langchain_community 导入
# 首先确保安装了 langchain-community: pip install -U langchain-community
try:
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    st.error("请先安装 langchain-community: `pip install -U langchain-community`")
    st.stop()

sys.path.append("../../")

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chat2Plot Demo (Aliyun)", page_icon=":robot:", layout="wide")
st.header("Chat2Plot Demo (Aliyun DashScope)")


def dynamic_install(module):
    sleep_time = 30
    dependency_warning = st.warning(
        f"Installing dependencies, this takes {sleep_time} seconds."
    )
    subprocess.Popen([f"{sys.executable} -m pip install {module}"], shell=True)
    # wait for subprocess to install package before running your actual code below
    time.sleep(sleep_time)
    # remove the installing dependency warning
    dependency_warning.empty()


# https://python.plainenglish.io/how-to-install-your-own-private-github-package-on-streamlit-cloud-eb3aaed9b179  
try:
    from chat2plot import ResponseType, chat2plot
    from chat2plot.chat2plot import Chat2Vega
except ModuleNotFoundError:
    github_token = st.secrets["github_token"]
    dynamic_install(f"git+https://{github_token}@github.com/nyanp/chat2plot.git")


def initialize_logger():
    logger = logging.getLogger("root")
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]
    return True


if "logger" not in st.session_state:
    st.session_state["logger"] = initialize_logger()

# --- 修改 2: API 密钥输入字段 ---
api_key = st.text_input("Step1: Input your DashScope API-KEY", value="", type="password", key="dashscope_api_key")

# --- 修改 3: 添加模型选择下拉框 ---
# 你可以根据阿里云文档添加更多支持的模型
supported_models = ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-long"]
selected_model = st.selectbox("Step2: Select Qwen Model", options=supported_models, index=0, key="selected_model")

csv_file = st.file_uploader("Step3: Upload csv file", type={"csv"})

if api_key and csv_file:
    # --- 修改 4: 设置阿里云 API 密钥环境变量 ---
    os.environ["DASHSCOPE_API_KEY"] = ''

    df = pd.read_csv(csv_file)

    st.write(df.head())

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    st.subheader("Chat")

    def initialize_c2p():
        # --- 修改 5: 创建并配置 ChatOpenAI 实例 ---
        chat_model = ChatOpenAI(
            model=st.session_state["selected_model"], # 使用用户选择的模型
            temperature=0,
            openai_api_key=os.environ.get("DASHSCOPE_API_KEY"), # 使用环境变量中的密钥
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1" # 关键：设置阿里云 API 端点
        )
        
        # --- 修改 6: 初始化 chat2plot 时传入自定义的 chat_model ---
        # 强制 function_call 为 True，假设所选模型都支持函数调用
        # 你也可以设置为 "auto"，但需要修改 chat2plot 源码中的 _has_function_call_capability 来识别 qwen 模型
        st.session_state["chat"] = chat2plot(
            df,
            st.session_state["chart_format"],
            chat=chat_model, # 传入配置好的 chat_model
            function_call=True, # 强制启用函数调用以获得最佳效果
            verbose=True,
            description_strategy="head",
        )

    def reset_history():
        initialize_c2p()
        st.session_state["generated"] = []
        st.session_state["past"] = []

    with st.sidebar:
        chart_format = st.selectbox(
            "Chart format",
            ("simple", "vega"),
            key="chart_format",
            index=0,
            on_change=initialize_c2p,
        )

        st.button("Reset conversation history", on_click=reset_history)

    if "chat" not in st.session_state:
        initialize_c2p()

    c2p = st.session_state["chat"]

    chat_container = st.container()
    input_container = st.container()

    def submit():
        submit_text = st.session_state["input"]
        st.session_state["input"] = ""
        with st.spinner(text="Wait for LLM response..."):
            try:
                if isinstance(c2p, Chat2Vega):
                    res = c2p(submit_text, config_only=True)
                else:
                    res = c2p(submit_text, config_only=False, show_plot=False)
            except Exception as e:
                st.error(f"An error occurred during query processing: {e}")
                res = traceback.format_exc()
        st.session_state.past.append(submit_text)
        st.session_state.generated.append(res)

    def get_text():
        input_text = st.text_input("You: ", key="input", on_change=submit)
        return input_text

    with input_container:
        user_input = get_text()

    if st.session_state["generated"]:
        with chat_container:
            for i in range(
                len(st.session_state["generated"])
            ):  # range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

                res = st.session_state["generated"][i]

                if isinstance(res, str):
                    # something went wrong
                    st.error(res.replace("\n", "\n\n"))
                elif res.response_type == ResponseType.SUCCESS:
                    message(res.explanation, key=str(i))

                    col1, col2 = st.columns([2, 1])

                    with col2:
                        config = res.config
                        if isinstance(config, BaseModel):
                            # --- 修改点：适配 Pydantic v2 ---
                            # 使用 model_dump_json 并传入配置
                            # exclude_none 在 model_dump 中设置，然后序列化
                            try:
                                # 尝试使用 model_dump_json (Pydantic v2 推荐)
                                json_str = config.model_dump_json(indent=2)
                            except AttributeError:
                                # 如果 model_dump_json 不可用（不太可能在 v2，但以防万一）
                                # 回退到 model_dump 然后 json.dumps
                                json_str = json.dumps(
                                    config.model_dump(mode='json', exclude_none=True),
                                    indent=2
                                )
                            st.code(json_str, language="json")
                            # --- 修改结束 ---
                        else:
                            st.code(json.dumps(config, indent=2), language="json")
                    with col1:
                        if isinstance(res.figure, Figure):
                            st.plotly_chart(res.figure, use_container_width=True)
                        else:
                            # Vega-Lite charts might need the dataframe passed explicitly
                            st.vega_lite_chart(df, res.config, use_container_width=True)
                else:
                    st.warning(
                        f"Failed to render chart. Response Type: {res.response_type}. Last message: {getattr(res, 'conversation_history', [{'content': 'N/A'}])[-1]['content'] if hasattr(res, 'conversation_history') and res.conversation_history else 'N/A'}",
                        icon="⚠️",
                    )
                    # message(res.conversation_history[-1].content, key=str(i))
import json
import logging
import os
import subprocess
import sys
import time
import traceback

import pandas as pd
import streamlit as st
from plotly.graph_objs import Figure
from pydantic import BaseModel
from streamlit_chat import message

# --- 修改 1: 导入正确的 ChatOpenAI ---
# 注意：根据警告信息，应从 langchain_community 导入
# 首先确保安装了 langchain-community: pip install -U langchain-community
try:
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    st.error("请先安装 langchain-community: `pip install -U langchain-community`")
    st.stop()

sys.path.append("../../")

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chat2Plot Demo (Aliyun)", page_icon=":robot:", layout="wide")
st.header("Chat2Plot Demo (Aliyun DashScope)")


def dynamic_install(module):
    sleep_time = 30
    dependency_warning = st.warning(
        f"Installing dependencies, this takes {sleep_time} seconds."
    )
    subprocess.Popen([f"{sys.executable} -m pip install {module}"], shell=True)
    # wait for subprocess to install package before running your actual code below
    time.sleep(sleep_time)
    # remove the installing dependency warning
    dependency_warning.empty()


# https://python.plainenglish.io/how-to-install-your-own-private-github-package-on-streamlit-cloud-eb3aaed9b179  
try:
    from chat2plot import ResponseType, chat2plot
    from chat2plot.chat2plot import Chat2Vega
except ModuleNotFoundError:
    github_token = st.secrets["github_token"]
    dynamic_install(f"git+https://{github_token}@github.com/nyanp/chat2plot.git")


def initialize_logger():
    logger = logging.getLogger("root")
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]
    return True


if "logger" not in st.session_state:
    st.session_state["logger"] = initialize_logger()

# --- 修改 2: API 密钥输入字段 ---
api_key = st.text_input("Step1: Input your DashScope API-KEY", value="", type="password", key="dashscope_api_key")

# --- 修改 3: 添加模型选择下拉框 ---
# 你可以根据阿里云文档添加更多支持的模型
supported_models = ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-long"]
selected_model = st.selectbox("Step2: Select Qwen Model", options=supported_models, index=0, key="selected_model")

csv_file = st.file_uploader("Step3: Upload csv file", type={"csv"})

if api_key and csv_file:
    # --- 修改 4: 设置阿里云 API 密钥环境变量 ---
    os.environ["DASHSCOPE_API_KEY"] = ''

    df = pd.read_csv(csv_file)

    st.write(df.head())

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    st.subheader("Chat")

    def initialize_c2p():
        # --- 修改 5: 创建并配置 ChatOpenAI 实例 ---
        chat_model = ChatOpenAI(
            model=st.session_state["selected_model"], # 使用用户选择的模型
            temperature=0,
            openai_api_key=os.environ.get("DASHSCOPE_API_KEY"), # 使用环境变量中的密钥
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1" # 关键：设置阿里云 API 端点
        )
        
        # --- 修改 6: 初始化 chat2plot 时传入自定义的 chat_model ---
        # 强制 function_call 为 True，假设所选模型都支持函数调用
        # 你也可以设置为 "auto"，但需要修改 chat2plot 源码中的 _has_function_call_capability 来识别 qwen 模型
        st.session_state["chat"] = chat2plot(
            df,
            st.session_state["chart_format"],
            chat=chat_model, # 传入配置好的 chat_model
            function_call=True, # 强制启用函数调用以获得最佳效果
            verbose=True,
            description_strategy="head",
        )

    def reset_history():
        initialize_c2p()
        st.session_state["generated"] = []
        st.session_state["past"] = []

    with st.sidebar:
        chart_format = st.selectbox(
            "Chart format",
            ("simple", "vega"),
            key="chart_format",
            index=0,
            on_change=initialize_c2p,
        )

        st.button("Reset conversation history", on_click=reset_history)

    if "chat" not in st.session_state:
        initialize_c2p()

    c2p = st.session_state["chat"]

    chat_container = st.container()
    input_container = st.container()

    def submit():
        submit_text = st.session_state["input"]
        st.session_state["input"] = ""
        with st.spinner(text="Wait for LLM response..."):
            try:
                if isinstance(c2p, Chat2Vega):
                    res = c2p(submit_text, config_only=True)
                else:
                    res = c2p(submit_text, config_only=False, show_plot=False)
            except Exception as e:
                st.error(f"An error occurred during query processing: {e}")
                res = traceback.format_exc()
        st.session_state.past.append(submit_text)
        st.session_state.generated.append(res)

    def get_text():
        input_text = st.text_input("You: ", key="input", on_change=submit)
        return input_text

    with input_container:
        user_input = get_text()

    if st.session_state["generated"]:
        with chat_container:
            for i in range(
                len(st.session_state["generated"])
            ):  # range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

                res = st.session_state["generated"][i]

                if isinstance(res, str):
                    # something went wrong
                    st.error(res.replace("\n", "\n\n"))
                elif res.response_type == ResponseType.SUCCESS:
                    message(res.explanation, key=str(i))

                    col1, col2 = st.columns([2, 1])

                    with col2:
                        config = res.config
                        if isinstance(config, BaseModel):
                            # --- 修改点：适配 Pydantic v2 ---
                            # 使用 model_dump_json 并传入配置
                            # exclude_none 在 model_dump 中设置，然后序列化
                            try:
                                # 尝试使用 model_dump_json (Pydantic v2 推荐)
                                json_str = config.model_dump_json(indent=2)
                            except AttributeError:
                                # 如果 model_dump_json 不可用（不太可能在 v2，但以防万一）
                                # 回退到 model_dump 然后 json.dumps
                                json_str = json.dumps(
                                    config.model_dump(mode='json', exclude_none=True),
                                    indent=2
                                )
                            st.code(json_str, language="json")
                            # --- 修改结束 ---
                        else:
                            st.code(json.dumps(config, indent=2), language="json")
                    with col1:
                        if isinstance(res.figure, Figure):
                            st.plotly_chart(res.figure, use_container_width=True)
                        else:
                            # Vega-Lite charts might need the dataframe passed explicitly
                            st.vega_lite_chart(df, res.config, use_container_width=True)
                else:
                    st.warning(
                        f"Failed to render chart. Response Type: {res.response_type}. Last message: {getattr(res, 'conversation_history', [{'content': 'N/A'}])[-1]['content'] if hasattr(res, 'conversation_history') and res.conversation_history else 'N/A'}",
                        icon="⚠️",
                    )
                    # message(res.conversation_history[-1].content, key=str(i))
