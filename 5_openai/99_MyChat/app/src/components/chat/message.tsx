import { mergeClasses } from "@fluentui/react-components";
import { useMessageStyles } from "./message.styles";

enum MessageRole {
  System = "system",
  Assistant = "assistant",
  User = "user",
}

export const Message = (props: any) => {
  const classes = useMessageStyles();
  return (
    <div
      className={mergeClasses(
        classes.container,
        props.data.role === MessageRole.System && classes.systemMessage,
        props.data.role === MessageRole.Assistant && classes.assistantMessage,
        props.data.role === MessageRole.User && classes.userMessage
      )}
    >
      {props.data.content}
    </div>
  );
};
