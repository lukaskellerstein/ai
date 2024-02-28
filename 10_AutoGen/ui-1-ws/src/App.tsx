import { Chat } from "@components/chat/chat";
import { SideBar } from "@components/sidebar/sidebar";
import { useClasses } from "./App.styles";

function App() {
  const classes = useClasses();

  return (
    <div className={classes.app}>
      <SideBar />
      <Chat />
    </div>
  );
}

export default App;
