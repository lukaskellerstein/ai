import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    file_location="./",
)

print("config_list: ", config_list)

llm_config = {
    "config_list": config_list,
    "temperature": 0,
}

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "temp_dir_0"},
)


# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""
Can you create a simple React application that is using FluentUI v9.0.0 and has a button that says "Click me"? After clicking on the button it should show an alert with the text "Hello world!".
""",
)
