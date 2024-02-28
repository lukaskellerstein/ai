import { makeStyles, shorthands } from "@fluentui/react-components";

export const useClasses = makeStyles({
  chat: {
    height: "100%",
    display: "flex",
    flexDirection: "column",
    flexGrow: 1,
  },
  messagePane: {
    display: "flex",
    flexDirection: "column",
    flexGrow: 1,
    overflowY: "auto",
    ...shorthands.padding("10px"),
  },
  newMessage: {
    width: "100%",
    display: "flex",
    minHeight: "50px",
    alignItems: "center",
  },
  messageInput: {
    flexGrow: 1,
  },
  sendButton: {
    minWindth: "85px",
    ...shorthands.margin(0, "7px"),
  },
});
