import { Button, Textarea } from "@fluentui/react-components";
import { Send24Regular } from "@fluentui/react-icons";
import { useEffect, useRef, useState } from "react";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { useClasses } from "./chat.styles";
import { Message } from "./message";

type Message = {
  role: string;
  content: string;
};

export const Chat = () => {
  const classes = useClasses();

  const inputRef = useRef<HTMLTextAreaElement>(null);

  const [messages, setMessages] = useState<Message[]>([]);

  const WS_URL = "ws://127.0.0.1:8000";
  const { sendJsonMessage, lastJsonMessage, readyState } = useWebSocket(
    WS_URL,
    {
      share: false,
      shouldReconnect: () => true,
    }
  );

  useEffect(() => {
    sendJsonMessage({ action: "register", payload: "web_client" });
  }, []);

  // Run when a new WebSocket message is received (lastJsonMessage)
  useEffect(() => {
    if (lastJsonMessage) {
      const message =
        typeof lastJsonMessage === "string"
          ? lastJsonMessage
          : JSON.stringify(lastJsonMessage);

      console.log(`Got a new message: ${message}`);

      setMessages((prevMessages) => {
        return [...prevMessages, { role: "assistant", content: message }];
      });
    }
  }, [lastJsonMessage]);

  const sendMessage = async () => {
    const message = inputRef.current?.value as string;
    console.log("send message", message);

    setMessages((prevMessages) => {
      return [...prevMessages, { role: "user", content: message }];
    });

    // SEND TO WS
    if (readyState === ReadyState.OPEN) {
      sendJsonMessage({
        action: "send_message",
        payload: {
          sender: "web_client",
          receivers: ["init_chat"],
          message: message,
        },
      });
    }
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
