import { FluentProvider, teamsLightTheme } from "@fluentui/react-components";
import { createRoot } from "react-dom/client";
import App from "./App";

const domNode = document.getElementById("root");
const root = createRoot(domNode as any);

root.render(
  <FluentProvider theme={teamsLightTheme} style={{ height: "100%" }}>
    <App />
  </FluentProvider>
);
