import { Button, Textarea } from "@fluentui/react-components";
import { Send24Regular } from "@fluentui/react-icons";
import { useEffect, useRef, useState } from "react";
import { getCurrentDateTime } from "../../helper";
import { NatsService } from "../../services/nats-service";
import { useClasses } from "./chat.styles";
import { Message } from "./message";

type Message = {
  role: string;
  content: string;
};

var environment = {
  brokerUrl: "127.0.0.1:8089",
  apiUrl: "localhost:XXXX",
  tenantId: "companyABC",
  userId: "daniel_superman",
  teamId: "my_team",
};

export const Chat = () => {
  const classes = useClasses();

  const inputRef = useRef<HTMLTextAreaElement>(null);

  const [messages, setMessages] = useState<Message[]>([]);

  const natsService = new NatsService(`ws://${environment.brokerUrl}`);

  useEffect(() => {
    (async () => {
      await natsService.connect();
    })();
  });

  const sendMessage = async () => {
    const message = inputRef.current?.value as string;
    console.log("send message", message);

    setMessages((prevMessages) => {
      return [...prevMessages, { role: "user", content: message }];
    });

    natsService.publish(
      `${environment.tenantId}.${environment.userId}.${
        environment.teamId
      }.${getCurrentDateTime()}.groupchat_manager`,
      {
        sender: "real_user",
        action: "init_chat",
        payload: {
          message: message,
        },
      }
    );
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
