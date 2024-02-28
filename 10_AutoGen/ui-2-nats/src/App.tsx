import { Chat } from "@components/chat/chat";
import { SideBar } from "@components/sidebar/sidebar";
import { useClasses } from "./App.styles";

import { useEffect } from "react";
import { NatsService } from "./services/nats-service";

function App() {
  const classes = useClasses();

  const natsService = new NatsService("ws://localhost:8089");

  useEffect(() => {
    (async () => {
      await natsService.connect();

      const sub = natsService.subscribe("tenantId.userId.chatId.web_client");
      (async () => {
        if (sub) {
          for await (const m of sub) {
            console.log(
              `[${sub.getProcessed()}]: ${natsService.decode(m.data)}`
            );
          }
          console.log("subscription closed");
        } else {
          console.log("subscription not ready");
        }
      })();
    })();
  });

  return (
    <div className={classes.app}>
      <SideBar />
      <Chat />
    </div>
  );
}

export default App;
