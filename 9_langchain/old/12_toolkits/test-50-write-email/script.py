from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials

from langchain.callbacks.human import HumanApprovalCallbackHandler


# ---------------------------
# HUMAN REVIEW
# ---------------------------
def _should_check(serialized_obj: dict) -> bool:
    print(serialized_obj)
    return (
        serialized_obj.get("name") == "send_gmail_message"
        or serialized_obj.get("name") == "create_gmail_draft"
    )


def _approve(_input: str) -> bool:
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + _input + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


callbacks = [HumanApprovalCallbackHandler(should_check=_should_check, approve=_approve)]

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# ---------------------------
# Can review scopes here https://developers.google.com/gmail/api/auth/scopes
# For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

tools = toolkit.get_tools()
for tool in tools:
    print(tool.name)

# ---------------------------
# AGENT
# ---------------------------
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# ---------------------------
# TEXT of EMAIL
# ---------------------------
text = """
    Role: You are an AI that help people writing their emails.

    User: Write a email subject and text of email for my wife (anna.kellerstein@gmail.com). My name is Lukas. The email content contain a three parts (sections, chapters). A lovely first chapter as an expression of my love and admiration to her. The Second chapter will contain a funny story about our future with our kids. The Third chapter will contain a funny story about our elderly days. Be creative and funny.
    Follow a structure in response:
    ---
    TO: email address of receiver
    SUBJECT: subject of email
    EMAIL: text of email
    ---
    AI:
    """

emailText = llm(text)

print("------ RESULT: Email text -------------")
print(emailText)
print("----------------------------------------")

# ---------------------------
# DRAFT EMAIL
# ---------------------------

# prompt = """
#     Create a gmail draft for my wife (anna.kellerstein@gmail.com).
#     Under no circumstances may you send the message.
#     Save the draft.

#     Follow the structure of the email below.

#     -----
#     {emailText}
#     -----
#     """.format(
#     emailText=emailText
# )

# result = agent.run(
#     prompt,
#     callbacks=callbacks,
# )
# print("------ RESULT: Draft email -------------")
# print(result)
# print("----------------------------------------")

# ---------------------------
# Find the EMAIL
# ---------------------------
# result = agent.run(
#     "Could you search in my drafts for the latest email? Reciver of the email is anna.kellerstein@gmail.com. Return me the draft id, subject and text of email.",
#     callbacks=callbacks,
# )
# print("------ RESULT: Find email -------------")
# print(result)
# print("----------------------------------------")

# ---------------------------
# SEND EMAIL => Does not work by ID, you can only send a new email
# ---------------------------

prompt = """
    Send the email by instructions below.

    -----
    {emailText}
    -----
    """.format(
    emailText=emailText
)

agent.run(
    prompt,
    callbacks=callbacks,
)
print("------ RESULT: Send email -------------")
# print(result2)
print("----------------------------------------")


# ---------------------------
# REPLY EMAIL => Does not exist !!
# ---------------------------
