import { Button, Textarea } from "@fluentui/react-components";
import { Send24Regular } from "@fluentui/react-icons";
import { useRef, useState } from "react";
import { useClasses } from "./chat.styles";
import { Message } from "./message";

type Message = {
  role: string;
  content: string;
};

export const Chat = () => {
  const classes = useClasses();

  const inputRef = useRef<HTMLTextAreaElement>(null);

  const [messages, setMessages] = useState<Message[]>([
    // { role: "system", content: "Welcome to the chatbot!" },
    // { role: "assistant", content: "Type a message to get started." },
    // {
    //   role: "user",
    //   content: "You can also click on the sidebar to switch chats.",
    // },
    // { role: "assistant", content: "Try asking the chatbot about the weather!" },
    // { role: "user", content: "Or ask it to tell you a joke!" },
    // { role: "assistant", content: "Or ask it to tell you a story!" },
    // { role: "user", content: "Or ask it to tell you a poem!" },
    // { role: "assistant", content: "Or ask it to tell you a fun fact!" },
    // { role: "user", content: "Or ask it to tell you a quote!" },
    // {
    //   role: "assistant",
    //   content:
    //     "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    // },
    // { role: "user", content: "Or ask it to tell you a tongue twister!" },
    // {
    //   role: "assistant",
    //   content: `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    //     Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.`,
    // },
  ]);

  const sendMessage = async () => {
    const message = inputRef.current?.value as string;
    console.log("send message", message);

    setMessages((prevMessages) => {
      return [...prevMessages, { role: "user", content: message }];
    });

    const response = await fetch("http://localhost:8000/send-message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ role: "user", content: message }),
    }).then((res) => res.json());

    console.log("response", response);

    setMessages((prevMessages) => {
      return [...prevMessages, response];
    });
  };

  const messagesUI = messages.map((message, index) => {
    return <Message key={index} data={message} />;
  });

  return (
    <div className={classes.chat}>
      <div className={classes.messagePane}>{messagesUI}</div>
      <div className={classes.newMessage}>
        <Textarea
          className={classes.messageInput}
          placeholder="type here..."
          resize="vertical"
          ref={inputRef}
        />
        <Button
          className={classes.sendButton}
          icon={<Send24Regular />}
          onClick={sendMessage}
        >
          Send
        </Button>
      </div>
    </div>
  );
};
